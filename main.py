"""
main.py —— FastAPI 服务（第 2 步最小版）

职责一句话：把 HTTP 请求翻译成 Python 调用，把结果翻译回 JSON。
它自己不产生任何智能——智能在 llm_client 和（将来的）内核里。

本版只有两个接口：
  GET  /           返回聊天页面
  POST /api/chat   收一句话，回一句话
"""

from contextlib import asynccontextmanager
from typing import Literal
import json
import uuid

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import config
import db
import kernel_agent
import kernel_rag
import vector_store


# ---- 生命周期：服务启动前跑一次 ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    config.check()          # 启动体检：key 没配等低级错误在这里拦下
    db.init_db()            # 建会话/消息表（IF NOT EXISTS，重复启动安全）
    vector_store.build()    # 启动建库（已存在则跳过）：别让第一个用户等建库的十几秒
    yield


app = FastAPI(title="客服小助手", lifespan=lifespan)


# ---- 请求/响应的数据契约 ----
# Pydantic 模型 = 接口的"门卫"：
#   请求体不符合定义（空消息、超长、字段名写错）→ 根本进不了业务函数，直接 422
class ChatMessage(BaseModel):
    """历史消息的一条。role 只允许 user/assistant ——
    前端不许自称 system（否则可以冒充人设、篡改客服指令，这是真实攻击面）"""
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    # history 的前端最多收 12 条。
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)
    # 双内核切换：agent（默认，自己决定查不查/查什么）或 rag（每问必检索）
    mode: Literal["agent", "rag"] = "agent"
    # 会话 ID：前端带的就用它的（持久化靠它）；没带的开新会话
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    # 答案的"证据"：本回答检索到了知识库的哪几块，前端展示、用户可核对
    sources: list[dict] = []
    # Agent 模式的"过程"：每一轮调了什么工具、传了什么参数（可观测性）
    steps: list[dict] = []
    session_id: str = ""


def _resolve_session(req: ChatRequest) -> tuple[str, list[dict]]:
    """定会话 + 取历史：
    1. 前端带了 session_id → 继续那个会话；
    2. 没带 → 开新会话；
    3. 前端没带历史（比如刚刷新页面）→ 从 SQLite 捞回来——会话不怕刷新了。
    """
    sid = req.session_id or uuid.uuid4().hex[:8]
    history = [{"role": m.role, "content": m.content} for m in req.history]
    if not history:
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in db.load_messages(sid, limit=12)
            if m["role"] in ("user", "assistant")
        ]
    return sid, history


# ---- 核心接口（非流式）----
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid, history = _resolve_session(req)
    try:
        if req.mode == "rag":
            result = kernel_rag.answer(req.message, history=history)
        else:
            result = kernel_agent.run(req.message, history=history)
    except Exception as e:
        # 工程防护：模型超时/限流、检索失败、工具崩溃……
        # 都不该把 500 甩给用户——给个体面的降级回答，错误留在服务端日志里
        print(f"[ERROR] /api/chat: {type(e).__name__}: {e}")
        return ChatResponse(reply="抱歉，系统开小差了，请稍后重试或联系人工客服。", session_id=sid)

    # 落盘：用户问题 + 机器人回答（附件里带着来源/轨迹，刷新后还能展示）
    db.save_message(sid, "user", req.message)
    db.save_message(sid, "assistant", result["answer"], meta={
        "sources": result.get("sources", []),
        "steps": result.get("steps", []),
    })
    return ChatResponse(
        reply=result["answer"],
        sources=result.get("sources", []),
        steps=result.get("steps", []),
        session_id=sid,
    )


# ---- 核心接口（流式 SSE）----
@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE 流式：把内核的事件一个个推给前端。
    SSE 协议极简：每条消息是 `data: {json}` + 两个换行，文本流不中断。"""
    sid, history = _resolve_session(req)

    gen = (kernel_rag.answer_stream(req.message, history)
           if req.mode == "rag"
           else kernel_agent.run_stream(req.message, history))

    def sse():
        full_answer = ""
        sources, steps = [], []
        try:
            for ev in gen:
                if ev["type"] == "delta":
                    full_answer += ev["text"]
                elif ev["type"] == "sources":
                    sources = ev["sources"]
                elif ev["type"] == "step":
                    steps.append({k: v for k, v in ev.items() if k != "type"})
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            db.save_message(sid, "user", req.message)
            db.save_message(sid, "assistant", full_answer, meta={
                "sources": sources, "steps": steps,
            })
            yield f"data: {json.dumps({'type': 'done', 'session_id': sid}, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"[ERROR] /api/chat/stream: {type(e).__name__}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'text': '抱歉，系统开小差了，请稍后重试或联系人工客服。'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


# ---- 会话查询接口 ----
@app.get("/api/sessions")
def get_sessions():
    return {"sessions": db.list_sessions()}


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    """某个会话的完整消息（前端刷新后用来恢复对话）。"""
    msgs = db.load_messages(sid, limit=100)
    return {
        "session_id": sid,
        "messages": [
            {"role": m["role"], "content": m["content"], "meta": m["meta"]}
            for m in msgs
        ],
    }


# ---- 聊天页面 ----
@app.get("/")
def index():
    # 把之前写好的前端页面作为首页返回
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.settings.HOST,
        port=config.settings.PORT,
    )
