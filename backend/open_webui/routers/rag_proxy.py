from fastapi import APIRouter, HTTPException, Request
import os, time, requests

router = APIRouter()
RAG_PROXY_URL = os.getenv("RAG_PROXY_URL", "http://host.docker.internal:8080")
RAG_PROXY_API_KEY = os.getenv("RAG_PROXY_API_KEY", "")

def _ask_rag(q: str, ns: str | None = None) -> str:
    h = {"X-API-Key": RAG_PROXY_API_KEY} if RAG_PROXY_API_KEY else {}
    p = {"question": q}
    if ns: p["namespace"] = ns
    r = requests.post(f"{RAG_PROXY_URL}/ask", json=p, headers=h, timeout=120)
    r.raise_for_status()
    return r.json().get("answer","")

@router.get("/v1/models")
async def models():
    return {"object":"list", "data":[{"id":"rag-proxy","object":"model"}]}

@router.post("/v1/chat/completions")
async def chat(body: dict, req: Request):
    msgs = body.get("messages") or []
    umsgs = [m for m in msgs if m.get("role")=="user" and m.get("content")]
    if not umsgs: raise HTTPException(400, "No user message")
    ns = (body.get("metadata") or {}).get("namespace")
    try:
        ans = _ask_rag(umsgs[-1]["content"], ns)
    except Exception as e:
        raise HTTPException(502, f"rag-proxy error: {e}")
    now = int(time.time())
    return {
        "id": f"chatcmpl-{now}", "object": "chat.completion", "created": now, "model": "rag-proxy",
        "choices": [{"index":0, "message":{"role":"assistant","content":ans}, "finish_reason":"stop"}],
        "usage": {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
    }
