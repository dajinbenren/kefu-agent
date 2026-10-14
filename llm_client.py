"""
llm_client.py —— 大模型客户端（第 2 步最小版）

职责一句话：别的文件把"用户问题"给我，我把"模型的回答"还回去。
中间怎么调千问、用什么 SDK、地址怎么拼——外界一概不用知道。

为什么单独一个文件而不写进 main.py：
  换模型供应商（千问→DeepSeek→本地模型）时，只动这一个文件，
  FastAPI 的路由、校验逻辑一行不改。
"""

from openai import OpenAI

from config import settings

# 注意：明明调的是千问，却写 OpenAI——千问提供 OpenAI 兼容模式，
# 接口格式一模一样，所以 SDK 用 openai 的，地址换成千问的。
#
# timeout=60：模型请求最多等 60 秒。
# 不设它，上游卡死时你的请求会挂到天荒地老——用户那边就是永远的"正在输入"。
client = OpenAI(api_key=settings.API_KEY, base_url=settings.BASE_URL, timeout=60.0)


def complete(messages: list[dict], temperature: float = 0.3) -> str:
    """最底层出口：谁来都带着拼好的 messages，我只管发出去把回答拿回来。

    chat()（普通多轮）和 kernel_rag（RAG，带检索资料）最后都走这里。
    """
    resp = client.chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def chat(message: str, history: list[dict] | None = None) -> str:
    """多轮对话：问题进，回答出（历史可选）。

    messages 的顺序就是模型"看到"的完整剧本：
        1. system —— 客服人设，永远排第一（整出戏的总基调）
        2. history —— 之前的问答对，让模型知道上文聊了什么
        3. user   —— 用户这轮的新问题，永远排最后（最显眼的位置）

    为什么新问题放最后：模型的注意力对"结尾"最敏感，
    当前问题紧贴着它，不容易被长历史稀释掉。
    """
    messages = [{"role": "system", "content": settings.SYSTEM_PROMPT}]
    if history:
        messages.extend(history)  # 历史已经是 [{role, content}] 格式，直接拼
    messages.append({"role": "user", "content": message})

    return complete(messages)


def chat_with_tools(messages: list[dict], tools: list[dict], temperature: float = 0.2):
    """Agent 专用出口：把工具说明书也发给模型，返回"原始消息对象"。

    和 complete() 的关键区别：
      1. 多传 tools + tool_choice="auto" —— 模型获得"可以选择调工具"的能力
      2. 返回的不是 str 而是消息对象——因为它有两种可能：
         - content 有字 → 模型直接回答了（不需要工具）
         - tool_calls 有值 → 模型请求调工具（需要后端执行并回填）
         就是这个"不确定"，逼出了 agent loop。
    """
    resp = client.chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto = 模型自己决定调不调
        temperature=temperature,
    )
    return resp.choices[0].message


def complete_stream(messages: list[dict], temperature: float = 0.3):
    """流式出口：不等全部生成完，模型吐一个字我就 yield 一个字。

    stream=True 时，API 返回的是一个"碎块流"：
    每个 chunk 只带新增的一小段文字（delta.content），
    拼起来才是完整回答。用户侧看到的效果 = 逐字打字。
    """
    resp = client.chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in resp:
        # 坑：流式的结尾 chunk 可能不带内容（choices 为空，只有用量统计），
        # 不判空直接 choices[0] 会 IndexError——真实踩到过的流式收尾坑
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


def embed(texts: list[str]) -> list[list[float]]:
    """把一批文本变成一批向量（每条 1024 个浮点数，config.EMBED_DIM）。

    两个 kb-site 里真踩过的坑，这里提前挡掉：
      1. 千问 embedding 单批最多 10 条，直接全丢过去报 400 → 自动分批
      2. 分批后拼回来的顺序必须与输入严格一致——上层靠这个顺序对齐原文，
         顺序错了不报错、只是答案全乱，极难排查
    """
    BATCH = 10
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        resp = client.embeddings.create(
            model=settings.EMBED_MODEL,
            input=texts[i:i + BATCH],
        )
        out.extend(d.embedding for d in resp.data)
    return out
