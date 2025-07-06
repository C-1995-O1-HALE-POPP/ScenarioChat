# 一个额外的多轮对话实现，字面意义的使用两个llm互相询问。
# 按照官网api教程使用assistant携带上下文实现记忆力机制。

import os
import sys
import json
import argparse
import threading
import time

from http import HTTPStatus
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

import dashscope
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role  # 角色常量



from utils.prompt import promptChat
from utils.duplication_check import get_existing_data, get_existng_ids
prompt_chat = promptChat()
existing_ids = set() 

MAX_WORKERS = 8  # 总线程数
MAX_API_CONC = 16  # API并发数
MAX_RETRY = 200
SEMAPHORE = threading.Semaphore(MAX_API_CONC)

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
    enable_thinking: bool = False,
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
    for i in range(MAX_RETRY):
        try:
            with SEMAPHORE:
                # 调用 DashScope Generation API
                resp = dashscope.Generation.call(
                    model=model,
                    messages=message_objs,
                    result_format="message",  # 要求返回 OpenAI ChatMessage 结构
                    temperature=temperature,
                    stream=False,             # 需要流式可改 True
                    enable_thinking=enable_thinking,
                )
            if resp.status_code == HTTPStatus.OK:
                logger.info(f"调用成功: {model}, 消息数: {len(messages)}")
                break
        except Exception as e:
            logger.error(f"调用失败: {e}, 重试 {i + 1}/{MAX_RETRY}")
            time.sleep(1)
    return _extract_content(resp)

def should_continue(history: List[Dict[str, str]]) -> bool:
    """
    判断对话是否可以继续。
    :param history: 历史消息列表
    :return: True 表示可以继续对话，False 表示对话结束
    """
    # 生成判断 prompt
    judger_prompt = prompt_chat.generate_judger_prompt(history)
    
    # 调用 LLM 判断
    response = call_llm(
        model=Generation.Models.qwen_turbo,
        messages=[{"role": Role.USER, "content": judger_prompt}],
        temperature=0.3,
        enable_thinking=False,
    )
    
    # 解析判断结果
    return not response.strip().lower() == "false" 
# --------------------------------------------------------------------------------------
# 多轮对话核心
# --------------------------------------------------------------------------------------
def run_multi_turn_dialog(
    turns: int,
    init_user_prompt: str,
    user_system_prompt: str,
    assistant_system_prompt: str,
    user_followup_prompt: str,
    assistant_followup_prompt: str,
    user_model: str = Generation.Models.qwen_turbo,
    assistant_model: str = Generation.Models.qwen_plus,
    temperature: float = 0.7,
    enable_thinking: bool = False,
) -> List[Dict[str, str]]:
    """
    让 user_model 和 assistant_model 进行多轮对话。
    一轮 = (user → assistant)。
    :return: 完整聊天记录（list[dict]）
    """

    history: List[Dict[str, str]] = [{"role": Role.USER, "content": init_user_prompt}]

    user_system_prompt_msg:         List[Dict[str, str]] = [{"role": Role.SYSTEM, "content": user_system_prompt}]
    assistant_system_prompt_msg:    List[Dict[str, str]] = [{"role": Role.SYSTEM, "content": assistant_system_prompt}]
    user_followup_msg:              List[Dict[str, str]] = [{"role": Role.SYSTEM, "content": user_followup_prompt}]
    assistant_followup_msg:         List[Dict[str, str]] = [{"role": Role.SYSTEM, "content": assistant_followup_prompt}]

    # -------- ② 主循环 --------
    early_stop = False
    for _ in range(turns):
        # 助理回复
        assistant_reply = call_llm(
            assistant_model,
            messages= assistant_system_prompt_msg + history + assistant_followup_msg,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
        history.append({"role": Role.ASSISTANT, "content": assistant_reply})

        # 模拟用户追问
        user_followup = call_llm(
            user_model,
            messages= user_system_prompt_msg + history + user_followup_msg,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
        history.append({"role": Role.USER, "content": user_followup})
        if not should_continue(history):
            logger.warning("提前终止对话")
            early_stop = True
            break
    return history, early_stop, len(history)

def write_to_file(data: Dict, output_file: str = "dialogue.json"):
    """线程安全的增量写入"""
    lock = threading.Lock()
    try:
        with lock:
            # 读取现有数据
            existing = get_existing_data(output_file)
            
            # 追加新数据
            existing.append(data)
            
            # 写入文件
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"写入文件失败: {e}")

def generate_dialogue_for_entry(entry: dict, user_model: str, assistant_model: str,
                                turns: int, temperature: float, enable_thinking: bool) -> Optional[dict]:
    if entry["id"] in existing_ids:
        logger.info(f"Skipping existing entry with id: {entry['id']}")
        return None

    config = entry["config"]
    topics, goal, strategy = config["topics"], config["goal"], config["strategy"]
    
    for scene in entry["scene"]:
        background, preference, question = scene["background"], scene["preference"], scene["question"]
        
        user_system_prompt = prompt_chat.generate_user_init_prompt(background, preference)
        user_followup_prompt = prompt_chat.generate_user_followup_prompt()
        assistant_system_prompt = prompt_chat.generate_assistant_init_prompt(topics, goal, strategy, background)
        assistant_followup_prompt = prompt_chat.generate_assistant_followup_prompt()
        user_init_prompt = preference + question

        result, early_stop, length = run_multi_turn_dialog(
            turns=turns,
            init_user_prompt=user_init_prompt,
            user_system_prompt=user_system_prompt,
            assistant_system_prompt=assistant_system_prompt,
            user_followup_prompt=user_followup_prompt,
            assistant_followup_prompt=assistant_followup_prompt,
            user_model=user_model,
            assistant_model=assistant_model,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
        scene["dialogue"] = result
        scene["early_stop"] = early_stop
        scene["length"] = length
    return entry

def run_concurrent_dialogue_generation(data_path: str, output_path: str, **kwargs):
    global existing_ids
    existing_ids = get_existng_ids(output_file=output_path)
    prompts = json.load(open(data_path, "r", encoding="utf-8"))

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(generate_dialogue_for_entry, entry, **kwargs): entry["id"]
            for entry in prompts if entry["id"] not in existing_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="生成对话"):
            try:
                result = future.result()
                if result:
                    write_to_file(result, output_file=output_path)
                    results.append(result)
            except Exception as e:
                logger.error(f"生成失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="Qwen Multi-Agent Chat (DashScope)")
    parser.add_argument("--data", type=str, default="backgrounds.json", help="背景数据文件路径")
    parser.add_argument("--turns", type=int, default=3, help="对话轮数（user+assistant 为 1 轮）")
    parser.add_argument("--user_model", type=str, default="qwen-turbo", help="用户模型名")
    parser.add_argument("--assistant_model", type=str, default="qwen-plus", help="助理模型名")
    parser.add_argument("--test", action='store_true', help="是否为测试模式（仅运行一次对话）")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度（0-2）")
    parser.add_argument("--enable_thinking", action='store_true', help="是否启用思考模式（模型可以进行思考）")
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

    if not os.path.exists("logs"):
        os.makedirs("logs")
    if not os.path.exists("results"):
        os.makedirs("results")
    os.makedirs("results/multiturn_dialogue", exist_ok=True)
    output_path = '''results/multiturn_dialogue/dialogue_''' + \
        f'''{args.user_model}_{args.assistant_model}_{args.turns}turns''' + \
        f'''_temperature{args.temperature}{"_thinking" if args.enable_thinking else ""}.json'''
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        f"logs/multiturn_dialogue_{time.strftime('%Y-%m-%d@%H:%M:%S')}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
        enqueue=True,
    )
    logger.info("开始多轮对话生成")
    logger.info(f"使用用户模型: {user_model}, 助理模型: {assistant_model}")
    logger.info(f"对话轮数: {args.turns}, 温度: {args.temperature}, 启用思考模式: {args.enable_thinking}")

    run_concurrent_dialogue_generation(
        data_path=args.data,
        output_path=output_path,
        user_model=user_model,
        assistant_model=assistant_model,
        turns=args.turns,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
    )


if __name__ == "__main__":
    # 若未配置 API-KEY，脚本直接退出
    if not os.getenv("DASHSCOPE_API_KEY"):
        sys.exit("❌  请先设置环境变量 DASHSCOPE_API_KEY，再运行此脚本！")
    main()
