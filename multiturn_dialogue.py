# 一个额外的多轮对话实现，字面意义的使用两个llm互相询问。
# 按照官网api教程使用assistant携带上下文实现记忆力机制。

import os
import sys
import argparse
from http import HTTPStatus
from typing import List, Dict, Any

import dashscope
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role  # 角色常量
from loguru import logger

# --------------------------------------------------------------------------------------
# 通用 LLM 调用封装
# --------------------------------------------------------------------------------------
def _extract_content(resp: Any) -> str:
    """
    从 DashScope 的响应对象中抽取 assistant 内容。
    同时兼容 prompt / messages 两种调用形态。
    """
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"DashScope error {resp.status_code}: {resp.message}")

    out = resp.output

    # ① prompt 方式返回纯字符串
    if isinstance(out, str):
        return out.strip()

    # ② messages 方式 result_format='message'，返回 dict
    if isinstance(out, dict):
        try:
            return out["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            pass

    # ③ 其它意外格式 —— 退化为 str(out)
    return str(out).strip()


def call_llm(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
) -> str:
    """
    单轮调用 DashScope Generation API（同步，非流式）。
    :param model:  Generation.Models.<model_name>  或直接填字符串模型名
    :param messages: OpenAI Chat 格式的历史消息
    :return: assistant 回复文本
    """
    # Convert list of dicts to list of Message objects
    from dashscope.api_entities.dashscope_response import Message
    message_objs = [Message(role=m["role"], content=m["content"]) for m in messages]
    logger.debug(f"Calling model: {model}, with messages: {message_objs}")
    resp = Generation.call(
        model=model,
        messages=message_objs,
        result_format="message",   # 要求返回 OpenAI ChatMessage 结构
        temperature=temperature,
        stream=False,              # 需要流式可改 True
    )
    return _extract_content(resp)


# --------------------------------------------------------------------------------------
# 多轮对话核心
# --------------------------------------------------------------------------------------
def run_multi_turn_dialog(
    turns: int,
    background: str,
    init_user_prompt: str | None = None,
    user_model: str = Generation.Models.qwen_turbo,
    assistant_model: str = Generation.Models.qwen_plus,
    temperature: float = 0.7,
) -> List[Dict[str, str]]:
    """
    让 user_model 和 assistant_model 进行多轮对话。
    一轮 = (user → assistant)。
    :return: 完整聊天记录（list[dict]）
    """
    # 全局 System 背景
    system_msg = {"role": Role.SYSTEM, "content": background}
    history: List[Dict[str, str]] = [system_msg]

    # -------- ① 首句生成或注入 --------
    if init_user_prompt:
        history.append({"role": Role.USER, "content": init_user_prompt})
    else:
        # 引导 Qwen-Turbo 说出第一句“学生发问”
        primer = {
            "role": Role.SYSTEM,
            "content": "你正在扮演一名本科生，请说出第一句对话。只返回用户的话。",
        }
        user_first = call_llm(
            user_model,
            messages=[system_msg, primer],
            temperature=temperature,
        )
        history.append({"role": Role.USER, "content": user_first})

    # -------- ② 主循环 --------
    for _ in range(turns):
        # 助理回复
        assistant_reply = call_llm(
            assistant_model,
            messages=history,
            temperature=temperature,
        )
        history.append({"role": Role.ASSISTANT, "content": assistant_reply})

        # 模拟用户追问
        user_followup = call_llm(
            user_model,
            messages=history + [
                {
                    "role": Role.SYSTEM,
                    "content": "请继续扮演用户，用口语方式回应上面助理的回答，只返回下一句。",
                }
            ],
            temperature=temperature,
        )
        history.append({"role": Role.USER, "content": user_followup})

    return history


# --------------------------------------------------------------------------------------
# CLI / Demo
# --------------------------------------------------------------------------------------
def _cli() -> None:
    parser = argparse.ArgumentParser(description="Qwen Multi-Agent Chat (DashScope)")
    parser.add_argument("--turns", type=int, default=3, help="对话轮数（user+assistant 为 1 轮）")
    parser.add_argument(
        "--background",
        type=str,
        default=(
            "场景：讨论如何在 FPGA 项目中实现 DDR3 帧缓冲读写与双缓冲。\n"
            "模拟用户：大三学生，略懂但仍有疑惑；"
            "助理：资深硬件工程师，需要循序渐进地解答并穿插代码示例与原理说明。"
        ),
        help="对话背景设定（System Prompt）",
    )
    parser.add_argument("--user_model", type=str, default="qwen-turbo", help="用户模型名")
    parser.add_argument("--assistant_model", type=str, default="qwen-plus", help="助理模型名")
    args = parser.parse_args()

    # 允许字符串或 Generation.Models 枚举
    user_model = (
        getattr(Generation.Models, args.user_model)
        if hasattr(Generation.Models, args.user_model)
        else args.user_model
    )
    assistant_model = (
        getattr(Generation.Models, args.assistant_model)
        if hasattr(Generation.Models, args.assistant_model)
        else args.assistant_model
    )
    logger.info(f"Using User Model: {user_model}, Assistant Model: {assistant_model}")

    dialog = run_multi_turn_dialog(
        turns=args.turns,
        background=args.background,
        init_user_prompt=None,
        user_model=user_model,
        assistant_model=assistant_model,
    )

    for msg in dialog:
        role = "👤User" if msg["role"] == Role.USER else "🤖Assistant"
        print(f"\n[{role}]: {msg['content']}")


if __name__ == "__main__":
    # 若未配置 API-KEY，脚本直接退出
    if not os.getenv("DASHSCOPE_API_KEY"):
        sys.exit("❌  请先设置环境变量 DASHSCOPE_API_KEY，再运行此脚本！")
    _cli()
