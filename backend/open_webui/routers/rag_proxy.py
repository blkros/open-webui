# open_webui/routers/rag_proxy.py
from fastapi import APIRouter, HTTPException, Request
import os, time
import httpx  # ## 변경: requests 대신 httpx만 사용 (비동기 호출)
from fastapi.responses import JSONResponse

router = APIRouter()

# ## 참고: 실제 값은 req.app.state.config.*에서 읽습니다.
#    아래는 (최초 기동/오프라인 등) 최후 보루용 기본값입니다.
RAG_PROXY_URL = os.getenv("RAG_PROXY_URL", "http://host.docker.internal:8080")
RAG_PROXY_API_KEY = os.getenv("RAG_PROXY_API_KEY", "")


def _forward_headers(req: Request) -> dict:
    """
    Open WebUI가 앞단에서 넣어주는 사용자 컨텍스트 헤더 전달용.
    """
    pass_headers: dict[str, str] = {}

    # ## 기존 그대로 전달할 사용자 컨텍스트 헤더들
    for h in [
        "x-openwebui-user-id",
        "x-openwebui-user-email",
        "x-openwebui-user-name",
    ]:
        v = req.headers.get(h)
        if v:
            pass_headers[h] = v

    # ## 변경: X-Forwarded-For는 '기존 값 + 클라이언트 IP'로 안전하게 누적
    client_ip = getattr(req.client, "host", None)
    existing = req.headers.get("x-forwarded-for")
    if client_ip:
        pass_headers["x-forwarded-for"] = (
            f"{existing}, {client_ip}" if existing else client_ip
        )

    return pass_headers


async def _call_proxy(req: Request, path: str):
    """
    /api/rag-proxy/* → 실제 rag-proxy 사이드카로 프록시하는 헬퍼
    """
    base = req.app.state.config.RAG_PROXY_URL
    timeout = req.app.state.config.RAG_PROXY_TIMEOUT
    api_key = req.app.state.config.RAG_PROXY_API_KEY

    url = f"{base.rstrip('/')}/{path.lstrip('/')}"
    method = req.method.upper()

    # ## 변경: Authorization + X-API-Key 둘 다 전달(서버 구현 차이 대비)
    hdrs = {
        "accept": "application/json",
        **_forward_headers(req),
    }
    if api_key:
        hdrs["authorization"] = f"Bearer {api_key}"
        hdrs["x-api-key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if method in ("GET", "DELETE"):
                resp = await client.request(
                    method, url, params=dict(req.query_params), headers=hdrs
                )
            else:
                # JSON 우선, 없으면 raw body
                body = await req.body()
                ctype = req.headers.get("content-type", "")
                if "application/json" in ctype.lower():
                    resp = await client.request(
                        method,
                        url,
                        content=body,
                        headers={"content-type": "application/json", **hdrs},
                    )
                else:
                    resp = await client.request(method, url, content=body, headers=hdrs)

        if resp.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        return JSONResponse(status_code=resp.status_code, content={"text": resp.text})
    except httpx.HTTPError as e:
        # ## 그대로 502로 감싸서 전달
        raise HTTPException(status_code=502, detail=f"RAG proxy upstream error: {e}") from e


@router.get("/_health")
async def health(req: Request):
    # 사이드카의 /health로 그대로 프록시
    return await _call_proxy(req, "/health")


# ## 변경: 동기 requests → 비동기 httpx + 앱 설정 일관 사용
async def _ask_rag(req: Request, q: str, ns: str | None = None) -> str:
    """
    rag-proxy의 /ask 엔드포인트를 호출하여 답변 텍스트를 받아옵니다.
    - 비동기(httpx.AsyncClient)
    - req.app.state.config.* 값 사용(운영 중 동적 반영)
    - Authorization + X-API-Key 헤더 모두 포함
    - 사용자 컨텍스트 헤더(X-OpenWebUI-*) 및 X-Forwarded-For 전달
    """
    base = req.app.state.config.RAG_PROXY_URL or RAG_PROXY_URL
    api_key = req.app.state.config.RAG_PROXY_API_KEY or RAG_PROXY_API_KEY
    timeout = req.app.state.config.RAG_PROXY_TIMEOUT

    url = f"{base.rstrip('/')}/ask"
    headers = {"accept": "application/json", **_forward_headers(req)}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
        headers["x-api-key"] = api_key

    payload = {"question": q}
    if ns:
        payload["namespace"] = ns

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)

    resp.raise_for_status()
    data = resp.json()
    return data.get("answer", "")


@router.get("/v1/models")
async def models():
    # OpenAI 호환 모델 목록 인터페이스
    return {"object": "list", "data": [{"id": "rag-proxy", "object": "model"}]}


@router.post("/v1/chat/completions")
async def chat(body: dict, req: Request):
    """
    OpenAI 호환 /v1/chat/completions → rag-proxy /ask로 포워드
    """
    msgs = body.get("messages") or []
    umsgs = [m for m in msgs if m.get("role") == "user" and m.get("content")]
    if not umsgs:
        raise HTTPException(400, "No user message")

    ns = (body.get("metadata") or {}).get("namespace")

    try:
        # ## 변경: 비동기 호출로 전환
        ans = await _ask_rag(req, umsgs[-1]["content"], ns)
    except Exception as e:
        # upstream 오류를 502로 래핑
        raise HTTPException(502, f"rag-proxy error: {e}")

    now = int(time.time())
    return {
        "id": f"chatcmpl-{now}",
        "object": "chat.completion",
        "created": now,
        "model": "rag-proxy",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": ans}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }