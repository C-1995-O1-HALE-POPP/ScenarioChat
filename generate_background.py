# 这里是全流程生成过程，背景 -> 问题 -> 对话流水线

from prompt import promptGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional
from tqdm import tqdm
from loguru import logger
import requests
import random
import os
import time
import argparse
import json
import re
import sys
import threading
import uuid
import argparse

# ===== 配置区 =====
API_KEY = os.getenv("DASHSCOPE_API_KEY")
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

MAX_WORKERS = 8  # 总线程数
MAX_API_CONC = 16  # API并发数
MAX_RETRY = 200
SEMAPHORE = threading.Semaphore(MAX_API_CONC)
output_file = "backgrounds.json"

model, thinking = "qwen-turbo", False
test = False
generator = promptGenerator(test=True)

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

def build_messages(user_prompt: str, system_prompt: str = None) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ] if system_prompt else [
        {"role": "user", "content": user_prompt}
    ]

def call_deepseek(messages: list[dict]) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "enable_thinking": thinking
    }
    with SEMAPHORE:
        resp = requests.post(API_URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def generate_questions_for_entry(entry: dict) -> Optional[dict]:
    """为单个entry生成问题和对话"""
    for _ in range(MAX_RETRY):
        try:
            question_prompt = generator.generate_question_prompt(
                background=entry["background"],
                preference=entry["preference"]
            )
            res = call_deepseek(build_messages(user_prompt=question_prompt))
            res = re.sub(r"```json\n(.*?)\n```", r"\1", res, flags=re.DOTALL)
            res = json.loads(res)
            entry["question"] = res["question"]
            entry["explanation"] = res["explanation"]
            entry["uuid"] = str(uuid.uuid4())
            entry["dialogue"] = generate_dialogue(entry["background"], entry["preference"] + entry["question"])
            return entry
        except requests.RequestException as e:
            if e.response and e.response.status_code == 429:
                logger.warning("API请求过于频繁，等待重试...")
                time.sleep(random.uniform(5, 10))
            else:
                logger.error(f"API请求失败: {e}")
                time.sleep(random.uniform(1, 3))
        except Exception as e:
            logger.warning(f"生成问题失败: {e}")
            time.sleep(random.uniform(1, 3))
    return None

def generate_scene_with_questions(prompt: dict) -> Optional[dict]:
    """生成场景及其所有问题"""
    config, content = prompt["config"], prompt["content"]
    scene = None
    
    # 生成场景
    for _ in range(MAX_RETRY):
        try:
            res = call_deepseek(build_messages(user_prompt=content))
            res = re.sub(r"```json\n(.*?)\n```", r"\1", res, flags=re.DOTALL)
            scene = json.loads(res)
            
            if not isinstance(scene, list):
                raise ValueError("场景响应不是列表")
                
            # 验证每个entry格式
            for entry in scene:
                if not isinstance(entry, dict) or "background" not in entry or "preference" not in entry:
                    raise ValueError(f"无效entry格式: {entry}")
            
            break

        except requests.RequestException as e:
            if e.response and e.response.status_code == 429:
                logger.warning("API请求过于频繁，等待重试...")
                time.sleep(random.uniform(5, 10))
            else:
                logger.error(f"API请求失败: {e}")
                time.sleep(random.uniform(1, 3))
        except Exception as e:
            logger.warning(f"生成场景失败: {e}")
            time.sleep(random.uniform(1, 5))
    
    if not scene:
        return None
    
    # 并发生成问题
    with ThreadPoolExecutor(max_workers=min(4, MAX_WORKERS//2)) as executor:
        futures = {executor.submit(generate_questions_for_entry, entry): entry for entry in scene}
        
        for future in as_completed(futures):
            entry = futures[future]
            try:
                result = future.result()
                if not result:
                    logger.error(f"为entry生成问题失败: {entry}")
            except Exception as e:
                logger.error(f"生成问题异常: {e}")
    
    return {
        "config": config,
        "scene": scene
    }

def generate_dialogue(scenario: str, question:str) -> list[dict]:
    """生成对话"""
    dialogue_prompt = generator.generate_dialogue_generation_prompt(scenario, question)
    for _ in range(MAX_RETRY):
        try:
            res = call_deepseek(build_messages(user_prompt=dialogue_prompt))
            res = re.sub(r"```json\n(.*?)\n```", r"\1", res, flags=re.DOTALL)
            dialogue = json.loads(res)
            if isinstance(dialogue, list) and all(
                isinstance(d, dict) and "role" in d and "content" in d for d in dialogue):
                return dialogue
            else:
                raise ValueError("对话格式不正确")
        except requests.RequestException as e:
            if e.response and e.response.status_code == 429:
                logger.warning("API请求过于频繁，等待重试...")
                time.sleep(random.uniform(5, 10))
            else:
                logger.error(f"API请求失败: {e}")
                time.sleep(random.uniform(1, 3))
        except Exception as e:
            logger.warning(f"生成对话失败: {e}")
            time.sleep(random.uniform(1, 5))
    raise RuntimeError("生成对话失败，已重试多次，但仍未成功")

def generate_background():
    """主生成函数"""
    prompts = list(generator.generate_all_background_prompt())
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(generate_scene_with_questions, prompt): prompt for prompt in prompts}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="生成场景"):
            prompt = futures[future]
            try:
                result = future.result()
                if result:
                    write_to_file(result)
                    results.append(result)
            except Exception as e:
                logger.error(f"处理prompt失败 '{prompt}': {e}")
    
    return results

def write_to_file(data: Dict):
    """线程安全的增量写入"""
    lock = threading.Lock()
    try:
        with lock:
            # 读取现有数据
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = []
            except (FileNotFoundError, json.JSONDecodeError):
                existing = []
            
            # 追加新数据
            existing.append(data)
            
            # 写入文件
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"写入文件失败: {e}")

def main():
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        f"logs/generate_background_{time.strftime('%Y-%m-%d@%H:%M:%S')}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
        enqueue=True,
    )
    parser = argparse.ArgumentParser(description="生成背景数据")
    parser.add_argument("--test", action="store_true", help="测试模式，仅生成一次")
    parser.add_argument("--output", type=str, default="backgrounds.json", help="输出文件路径")
    parser.add_argument("--model", type=str, default="qwen-turbo", help="使用的模型名称")
    parser.add_argument("--thinking", action="store_true", help="启用思考模式")
    args = parser.parse_args()
    global test, model, thinking, output_file
    test, model, thinking, output_file = args.test, args.model, args.thinking, args.output
    logger.info(f"测试模式: {test}, 使用模型: {model}, 思考模式: {thinking}, 输出文件: {output_file}")
    logger.info("开始生成背景数据...")
    generate_background()

if __name__ == "__main__":
    main()