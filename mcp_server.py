"""
mcp_server.py —— 把 kefu-agent 的"订单查询"封装成标准 MCP Server

目的：对应简历技能栏里的 MCP，让"MCP"不再是裸词。
- 用官方 mcp SDK 的 FastMCP 封装
- 默认 stdio transport（Claude Desktop / 各类 Agent 客户端可直接接）
- 设置环境变量 MCP_TRANSPORT=sse 可切换为 SSE transport

运行：
    python mcp_server.py                        # stdio（默认）
    MCP_TRANSPORT=sse python mcp_server.py     # SSE

说明：
- query_order 用与 kernel_agent.ORDERS 完全一致的模拟订单库，保证 MCP 工具
  与 Agent 内部工具行为一致。
- search_kb 走项目检索底座（vector_store），需要 .env 中的千问 key，
  故懒加载：不调用该工具时也能正常起服务。
"""

import os
import json

from mcp.server.fastmcp import FastMCP

# 与 kernel_agent.ORDERS 保持一致（模拟订单库，验收 Agent 是真查了还是编的）
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

mcp = FastMCP("kefu-order-mcp")


@mcp.tool()
def query_order(order_id: str) -> str:
    """按订单号查询订单状态、物流、金额。参数 order_id 格式：SF+13位数字。
    客户未提供订单号时不要猜测，应先索要。"""
    oid = order_id.strip().upper()
    order = ORDERS.get(oid)
    if order:
        return json.dumps({"order_id": oid, **order}, ensure_ascii=False)
    return f"订单号 {order_id} 不存在。请与客户核对订单号（格式：SF + 13位数字）。"


@mcp.tool()
def search_kb(keyword: str, top_k: int = 3) -> str:
    """检索官方知识库（退换货政策、配送、常见问题、产品资料）。
    凡涉及政策/时效/费用/流程的问题，优先用本工具，禁止凭记忆作答。"""
    from kernel_agent import _tool_search_kb  # 懒加载：需要向量库 + 千问 key
    return _tool_search_kb(keyword, top_k=top_k)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)
