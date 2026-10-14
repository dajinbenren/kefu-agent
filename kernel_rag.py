"""
kernel_rag.py —— RAG 内核：检索增强问答

职责一句话：用户问题 → 先去知识库捞原文 → 把原文和问题一起塞给模型
→ 模型"照着资料回答"而不是"凭记忆自由发挥"。

和普通聊天的本质区别（就多了一个动作）：
    普通聊天:  [人设, 历史, 问题]                    → 模型，靠自己的想象
    RAG:      [人设+防幻觉规则, 历史, 资料+问题]      → 模型，只准抄资料
"""

import config
import llm_client
import vector_store

# ---- 防幻觉规则：RAG 的安全底线 ----
# 它的作用不是让模型变聪明，而是给它一条"允许认输"的退路：
# 不告诉它"可以说不知道"，它就会在资料没答案时硬编一个——那才是真幻觉。
RAG_RULES = """
回答规范（必须遵守）：
1. 回答必须以【参考资料】为依据，资料来自星辰科技官方知识库。
2. 参考资料里没有明确写到的，如实告知"知识库中暂未提及该信息，建议联系人工客服"，
   绝不允许推测、不允许用你自己的常识补齐。
3. 涉及时限、金额、流程等具体数字时，必须与资料原文一致，禁止四舍五入或改写。
4. 资料中如有多条相关内容，全部整合后回答，不要只挑一条。
"""


def _prepare(question: str, history: list[dict] | None, top_k: int | None):
    """检索 + 装配的公共部分。answer()（非流式）和 answer_stream()（流式）共用。"""
    hits = vector_store.search(question, top_k=top_k)
    if not hits:
        return None, None
    # 【来源：xx 第x块】标签写进 prompt 有两个作用：
    #   a. 模型知道每段话的出处，引用时心里有数
    #   b. 原样返回给前端，用户能核对原文——RAG 的"可溯源"就落在这里
    context = "\n\n".join(
        f"【来源：{h['source']} 第{h['idx']}块】\n{h['text']}" for h in hits
    )
    messages = [
        {"role": "system", "content": config.settings.SYSTEM_PROMPT + "\n" + RAG_RULES},
    ]
    if history:
        messages.extend(history)
    # 资料和问题放同一条 user 消息里，且资料在前、问题在后：
    # 模型对结尾最敏感，问题贴着结尾，资料紧挨着问题
    messages.append({
        "role": "user",
        "content": f"【参考资料】\n{context}\n\n客户问题：{question}",
    })
    return hits, messages


def answer(question: str, history: list[dict] | None = None, top_k: int | None = None) -> dict:
    """RAG 问答（非流式）。返回 {"answer": str, "sources": [命中的块]}"""
    hits, messages = _prepare(question, history, top_k)
    if hits is None:
        return {
            "answer": "知识库暂时打不开，请稍后再试或联系人工客服。",
            "sources": [],
        }
    # 有资料可抄，温度再降一档
    reply = llm_client.complete(messages, temperature=0.2)
    return {
        "answer": reply,
        "sources": [
            {"source": h["source"], "idx": h["idx"], "rank": h["rank"]} for h in hits
        ],
    }


def answer_stream(question: str, history: list[dict] | None = None, top_k: int | None = None):
    """RAG 问答（流式）。逐事件产出：
       sources —— 本次命中的知识库块
       delta   —— 模型回答的一小段（前端拼出逐字打字效果）
    """
    hits, messages = _prepare(question, history, top_k)
    if hits is None:
        yield {"type": "delta", "text": "知识库暂时打不开，请稍后再试或联系人工客服。"}
        return
    yield {
        "type": "sources",
        "sources": [
            {"source": h["source"], "idx": h["idx"], "rank": h["rank"]} for h in hits
        ],
    }
    for piece in llm_client.complete_stream(messages, temperature=0.2):
        yield {"type": "delta", "text": piece}
