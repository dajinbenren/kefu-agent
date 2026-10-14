"""
kernel_agent.py —— Function Calling Agent 内核

职责一句话：让模型自己决定"要不要查知识库 / 要不要查订单"，
后端负责执行它请求的工具、把结果回填，循环直到它给出最终回答。

和 RAG 内核的区别：
    RAG:   每问必检索 → 资料塞 prompt → 单轮作答
    Agent: 模型看情况 → 请求调工具 → 后端执行 → 结果回填 → 模型可能再调
           （闲聊零检索；复杂问题可多步取证）

安全模型：模型只能"请求"调工具（返回工具名+参数字符串），
真正执行靠 FUNC_MAP 白名单查表——模型永远无法直接执行你的代码。
"""

import json

import config
import llm_client
import vector_store

# ============================================================
# 模拟订单库（真实业务里这里是数据库或订单系统的 API）
# 三个状态各一单，方便验收 Agent 是真查了还是编的
# ============================================================
ORDERS = {
    "SF202610130001": {
        "product": "星辰智能音箱Pro", "amount": 299,
        "status": "已签收", "logistics": "顺丰速运 SF1234567",
        "placed_at": "2026-10-11 20:15",
    },
    "SF202610130002": {
        "product": "星辰机械键盘K87", "amount": 459,
        "status": "已发货，运输中", "logistics": "中通快递 7834561234",
        "placed_at": "2026-10-12 09:30",
    },
    "SF202610130003": {
        "product": "星辰降噪耳机", "amount": 899,
        "status": "已打包，等待揽收", "logistics": "暂无运单号",
        "placed_at": "2026-10-14 10:05",
    },
}

# ============================================================
# 一、工具实现（真正干活的 Python 函数）
# ============================================================
_SOURCE_LOG: list[dict] = []  # 每次 run() 清空，记录本答检索到了哪些块（给前端展示）


def _tool_search_kb(keyword: str, top_k: int = 3) -> str:
    """查知识库 —— 就是把 kernel_rag 用的那个检索降级成了'工具'。
    RAG-as-Tool：同一套检索能力，Agent 内核复用，一行没重写。"""
    hits = vector_store.search(keyword, top_k=top_k)
    if not hits:
        return "知识库中没有找到相关内容。"
    for h in hits:
        if not any(s["source"] == h["source"] and s["idx"] == h["idx"] for s in _SOURCE_LOG):
            _SOURCE_LOG.append({"source": h["source"], "idx": h["idx"], "rank": h["rank"]})
    return "\n\n".join(f"【来源：{h['source']} 第{h['idx']}块】\n{h['text']}" for h in hits)


def _tool_query_order(order_id: str) -> str:
    """查订单 —— 模拟订单库查询。真实业务里这里会连数据库。"""
    oid = order_id.strip().upper()
    order = ORDERS.get(oid)
    if order:
        return json.dumps({"order_id": oid, **order}, ensure_ascii=False)
    return f"订单号 {order_id} 不存在。请与客户核对订单号（格式：SF + 13位数字）。"


# ============================================================
# 二、工具说明书（模型唯一能看到的"菜单"，JSON 契约）
# description 的措辞直接决定 Agent 靠不靠谱——写清楚"什么时候必须调"
# ============================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": (
                "检索星辰科技官方知识库（退换货政策、配送说明、常见问题、产品资料）。"
                "凡涉及公司政策、时效、费用、流程、产品使用的问题，必须先调用本工具，"
                "禁止凭自己的记忆回答政策类问题。找不到时可将 top_k 加大到 5 再试一次。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "检索关键词，用政策术语而非口语"},
                    "top_k": {"type": "integer", "description": "返回条数，默认 3"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": (
                "按订单号查询订单状态、物流、金额。参数 order_id 客户提供时才可调用；"
                "客户没有提供订单号时，禁止猜测订单号，应先礼貌地向客户索要。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "完整订单号，格式 SF+13位数字"},
                },
                "required": ["order_id"],
            },
        },
    },
]

# 白名单：模型喊出的工具名 → 真实函数。
# 模型只能点到表里有的，点不到的一律"未知工具"。
FUNC_MAP = {
    "search_kb": _tool_search_kb,
    "query_order": _tool_query_order,
}

# ============================================================
# 三、Agent 人设与用械规则
# ============================================================
AGENT_PROMPT = f"""{config.settings.SYSTEM_PROMPT}

工具使用规则：
1. 政策/时效/费用/流程/产品类问题 → 必须先调 search_kb，答案以检索结果为准，数字必须与资料一致。
2. 客户问自己的订单 → 调 query_order；没有订单号就先向客户索要，绝不猜测订单号。
3. 闲聊、问候、感谢 → 直接回答，不要调用任何工具。
4. 工具返回的资料里没有答案 → 如实告知"知识库中暂未提及该信息，建议联系人工客服"，不许编造。
5. 回答保持亲切简洁，一次只解决客户当前的问题。"""

MAX_TURNS = 5  # 防护1：轮次上限，防模型抽风无限循环烧钱


# ============================================================
# 四、Agent Loop 本体
# ============================================================
def run_stream(question: str, history: list[dict] | None = None):
    """生成器版 Agent：边跑边往外吐事件（SSE 的动力源）。
       step    —— 每次工具调用（前端实时显示轨迹）
       sources —— 最终答案前，汇总本答引用过哪些知识库块
       delta   —— 最终回答。Agent 是多步推理，答案在 loop 结束后才产出，
                  所以不像 RAG 那样逐字流——这是结构使然，不是偷懒。
    """
    _SOURCE_LOG.clear()
    cache: dict = {}          # 防护2：工具结果缓存——同一工具+同一参数不重复执行
    last_key = None           # 防护3：连续重复调用熔断的记录器
    repeats = 0

    messages = [{"role": "system", "content": AGENT_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    for turn in range(1, MAX_TURNS + 1):
        msg = llm_client.chat_with_tools(messages, TOOLS)

        # ---- 出口：模型不再请求工具 = 它觉得证据够了，直接给最终回答 ----
        if not msg.tool_calls:
            if _SOURCE_LOG:
                yield {"type": "sources", "sources": list(_SOURCE_LOG)}
            yield {"type": "delta", "text": msg.content or ""}
            return

        # 模型这轮的"请求"要先记进剧本，不然下一轮它看不到自己调过什么
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        # ---- 逐个执行它请求的工具 ----
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
            if key in cache:  # 缓存命中：不执行，直接复用结果
                result, cached = cache[key], True
            else:
                fn = FUNC_MAP.get(name)
                result = str(fn(**args)) if fn else f"未知工具：{name}"
                cache[key] = result
                cached = False

            # 熔断检测：连续两次"完全相同"的调用 = 模型卡住了
            repeats = repeats + 1 if key == last_key else 0
            last_key = key

            # 每调一个工具立刻吐事件，前端轨迹实时刷新
            yield {"type": "step", "turn": turn, "tool": name, "args": args, "cached": cached}

            if repeats >= 2:
                messages.append({
                    "role": "user",
                    "content": "你已连续多次用相同参数调用同一工具。请停止调用工具，基于已获取的资料直接给出最终答案。",
                })
                break  # 跳出工具循环，回去让模型收口

            # 灵魂一步：结果必须以 role=tool 回填，模型下一轮才看得见
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,  # tc 是 SDK 对象，属性访问不是 tc["id"]
                "content": result,
            })

    # ---- 轮次用尽兜底：强制收口，保证一定有最终回答 ----
    final = llm_client.complete(messages)
    if _SOURCE_LOG:
        yield {"type": "sources", "sources": list(_SOURCE_LOG)}
    yield {"type": "delta", "text": final}


def run(question: str, history: list[dict] | None = None) -> dict:
    """非流式包装：把流式事件攒成完整结果——/api/chat 用。
    逻辑只写一遍（在 run_stream），这里只做收集。"""
    answer, steps, sources = "", [], []
    for ev in run_stream(question, history):
        if ev["type"] == "delta":
            answer += ev["text"]
        elif ev["type"] == "step":
            steps.append({k: v for k, v in ev.items() if k != "type"})
        elif ev["type"] == "sources":
            sources = ev["sources"]
    return {"answer": answer, "steps": steps, "sources": sources}
