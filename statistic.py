#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计对话数据集中各场景类别的提前终止率
兼容 scene 列表内含多条子场景（如 20 条） 的数据结构
"""

import json
import sys
from collections import defaultdict

def load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def summarize(records):
    """
    records: list[dict]
        - item["config"]["topics"]  → 场景类别（五类之一）
        - item["scene"]             → list[dict]，每个 dict 含:
              • early_stop  (bool)  是否提前终止
              • length      (int)   对话轮数
    返回 stats: {类别: [场景数, 总轮次, 提前终止数]}
    """
    stats = defaultdict(lambda: [0, 0, 0])

    for item in records:
        category = item["config"]["topics"]
        for s in item["scene"]:                # 遍历 20 条子场景
            early  = bool(s.get("early_stop", False))
            length = int(s.get("length", 0))

            stats[category][0] += 1            # 场景数 +1
            stats[category][1] += (length - 1)/2       # 累积轮次
            stats[category][2] += early        # 累积提前终止
    return stats

def to_md(stats):
    header = "| 场景类别 | 场景数 | 平均轮次 | 提前终止次数 | 提前终止率 |"
    split  = "|---|---|---|---|---|"
    lines  = [header, split]

    total_scene = total_round = total_stop = 0
    for cat, (cnt, rounds, stop) in stats.items():
        avg_round = rounds / cnt if cnt else 0
        stop_rate = stop  / cnt if cnt else 0
        lines.append(
            f"| {cat} | {cnt} | {avg_round:.1f} | {stop} | {stop_rate:.0%} |"
        )
        total_scene += cnt
        total_round += rounds
        total_stop  += stop

    avg_all   = total_round / total_scene if total_scene else 0
    stop_all  = total_stop  / total_scene if total_scene else 0
    lines.append(
        f"| **合计** | **{total_scene}** | **{avg_all:.1f}** | "
        f"**{total_stop}** | **{stop_all:.0%}** |"
    )
    return "\n".join(lines)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "dialogue.json"
    stats = summarize(load(path))
    print(to_md(stats))

if __name__ == "__main__":
    main()
