#!/usr/bin/env python3
"""
analyze_calibration.py - 校准实验结果分析

对比同一 Judge 在【原版 Prompt】与【校准版 Prompt】下的判定差异，
并用人工核验真值评估两者的准确率。

用法:
  python analyze_calibration.py                 # 自动发现 calibration/ 下所有结果
  python analyze_calibration.py --judge deepseek-r1:14b
"""

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REVIEW_DIR = PROJECT_ROOT / "outputs" / "local_eval" / "reviews"
CALIB_DIR = PROJECT_ROOT / "outputs" / "local_eval" / "calibration"
PRED_FILE = PROJECT_ROOT / "outputs" / "local_eval" / "predictions" / "qwen2.5_7b.jsonl"

# ============================================================
# 人工核验真值（对照 50 题原文逐条判定）
#
# 判定依据：官方 ChineseSimpleQA 评分规则
#   A = 含参考答案要点且无矛盾信息（附加的非矛盾信息不影响）
#   B = 含与参考答案矛盾的事实陈述，或给出的实体/数值与参考答案不同
#   C = 未含参考答案要点、也无矛盾（拒答/不知道/截断）
#
# 关键事实已联网核实（见 HANDOVER.md 校准结论一节的引用链接）：
#   #16 卢梭《论人类不平等的起源和基础》确为 1755 年 4 月初版于阿姆斯特丹
#       → 模型答"1754年"与参考答案矛盾 → B
#   #20 1954 年被伊朗当局审判的是伊朗人民党（图德党）
#       → 模型给出"伊朗民族解放阵线"这一不同组织 → B
#   #42 书目数据库存储的是"二次文献"
#       → 模型答"元数据"，给出了与参考答案不同的概念 → B
#   #44 金斗瓮在广东、香港俗称"金塔" → 模型答"猪笼草" → B（已核实）
#   #50 释行真确为少林寺第三十二代嫡传弟子
#       → 模型称其"并非少林寺嫡传弟子、无代数" → 否认前提且与参考答案矛盾 → B
# ============================================================
HUMAN_GROUND_TRUTH = {
    # --- 明确判对 (A) ---
    5:  ("A", "先驱者10号 明确给出，附加信息不矛盾"),
    8:  ("A", "1883 给出"),
    9:  ("A", "2021年 给出"),
    11: ("A", "喻浩/喻皓 异体字，规则7"),
    27: ("A", "初级卵母细胞 给出（过程解释有误但核心答案正确，不构成矛盾）"),
    # --- 明确答错 (B) ---
    2:  ("B", "答 社会民主主义，未落到 社会主义"),
    16: ("B", "答 1754年出版，与 1755 矛盾（已核实）"),
    20: ("B", "答 伊朗民族解放阵线，应为 人民党（已核实）"),
    42: ("B", "答 元数据，应为 二次文献"),
    44: ("B", "答 猪笼草，应为 金塔（已核实）"),
    50: ("B", "否认前提，称其并非少林寺嫡传弟子（已核实）"),
    29: ("B", "答 日本生物环境化学学会，应为 NHK"),
    # --- 明确未尝试 (C) ---
    3:  ("C", "输出截断，无答案"),
    10: ("C", "明确说没有找到记录"),
    13: ("C", "明确说没有找到具体费用"),
}

JUDGE_FILES = {
    "deepseek-r1:14b": ("deepseek-r1_14b", "deepseek-r1:14b"),
    "deepseek-r1:7b": ("deepseek-r1_7b", "deepseek-r1:7b"),
    "qwen2.5:7b": ("qwen2.5_7b", "qwen2.5:7b"),
    "MichelRosselli/bonsai-27b": ("MichelRosselli_bonsai-27b", "MichelRosselli/bonsai-27b"),
}


def load_jsonl_map(path):
    """返回 (ordered_ids, {id: record})"""
    ids, recs = [], {}
    if not Path(path).exists():
        return ids, recs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            ids.append(o["id"])
            recs[o["id"]] = o
    return ids, recs


def score(grade_by_index):
    """用人工真值计算准确率，返回 (hit, total, errors)"""
    hit, total, errors = 0, 0, []
    for idx, human in HUMAN_GROUND_TRUTH.items():
        if idx not in grade_by_index:
            continue
        total += 1
        got = grade_by_index[idx]
        if got == human[0]:
            hit += 1
        else:
            errors.append((idx, human[0], got, human[1]))
    return hit, total, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", type=str, default=None,
                    help="只分析指定 judge（模型名）")
    args = ap.parse_args()

    order_ids, _ = load_jsonl_map(PRED_FILE)
    idx_of = {qid: i + 1 for i, qid in enumerate(order_ids)}

    # 发现校准结果文件
    calib_files = sorted(glob.glob(str(CALIB_DIR / "calibrated__*.jsonl")))
    if not calib_files:
        print(f"[ERR] 未找到校准结果: {CALIB_DIR / 'calibrated__*.jsonl'}")
        return

    safe_to_judge = {v[0]: k for k, v in JUDGE_FILES.items()}

    print("=" * 78)
    print("  JUDGE 校准实验分析  (原版 Prompt  vs  校准版 Prompt)")
    print("=" * 78)
    print(f"  人工核验题数: {len(HUMAN_GROUND_TRUTH)} / {len(order_ids)}")

    summary = []

    for cf in calib_files:
        stem = Path(cf).stem.replace("calibrated__", "")
        judge_model = safe_to_judge.get(stem, stem)
        if args.judge and judge_model != args.judge:
            continue

        _, cal_recs = load_jsonl_map(cf)
        base_file = REVIEW_DIR / f"qwen2.5_7b__judged_by__{stem}.jsonl"
        _, base_recs = load_jsonl_map(base_file)

        cal_grades = {idx_of[q]: r["grade"] for q, r in cal_recs.items() if q in idx_of}
        base_grades = {idx_of[q]: r["grade"] for q, r in base_recs.items() if q in idx_of}

        print(f"\n{'─' * 78}")
        print(f"  Judge: {judge_model}")
        print(f"{'─' * 78}")
        print(f"  原版结果: {base_file.name}  (n={len(base_grades)})")
        print(f"  校准结果: {Path(cf).name}  (n={len(cal_grades)})")

        for label, gr in (("原版", base_grades), ("校准", cal_grades)):
            if not gr:
                print(f"    [{label}] 无数据")
                continue
            cnt = Counter(gr.values())
            n = len(gr)
            print(f"    [{label}] A={cnt['A']} ({cnt['A']/n*100:.0f}%)  "
                  f"B={cnt['B']} ({cnt['B']/n*100:.0f}%)  "
                  f"C={cnt['C']} ({cnt['C']/n*100:.0f}%)")

        # 准确率（对照人工真值）
        for label, gr in (("原版", base_grades), ("校准", cal_grades)):
            hit, total, errors = score(gr)
            if total:
                print(f"\n    [{label} Prompt] 对照人工真值: {hit}/{total} 正确 "
                      f"(误判 {total-hit}, 误判率 {(total-hit)/total*100:.1f}%)")
                for idx, want, got, why in sorted(errors):
                    print(f"        #{idx:<3} 应为 {want}, 判为 {got}   [{why}]")

        # 原版 -> 校准 的变化
        if base_grades and cal_grades:
            changed = [(i, base_grades[i], cal_grades[i])
                       for i in sorted(cal_grades)
                       if i in base_grades and base_grades[i] != cal_grades[i]]
            print(f"\n    Prompt 变更引起的判定变化: {len(changed)} 题")
            for idx, b, c in changed:
                qid = order_ids[idx - 1]
                model_out = (cal_recs.get(qid) or {}).get("model_output", "")
                human = HUMAN_GROUND_TRUTH.get(idx)
                tag = ""
                if human:
                    tag = f"  ← 人工真值={human[0]}" + ("  ✓校准修正" if c == human[0] and b != human[0]
                                                    else "  ✗校准引入错误" if c != human[0] and b == human[0]
                                                    else "")
                print(f"        #{idx:<3} {b} → {c}{tag}")
                print(f"              {model_out[:90]!r}")

            # 校准后与原版仍不同 = 该 judge 对这两题口径变了但都没命中
            hit_b, tot, _ = score(base_grades)
            hit_c, _, _ = score(cal_grades)
            summary.append((judge_model, tot, hit_b, hit_c))

    # 总表
    if summary:
        print(f"\n{'=' * 78}")
        print("  汇总对比（按人工真值）")
        print(f"{'=' * 78}")
        print(f"  {'Judge':<30} {'原版正确':>10} {'校准正确':>10} {'变化':>8}")
        print(f"  {'-' * 62}")
        for judge, tot, hb, hc in summary:
            delta = hc - hb
            arrow = f"+{delta}" if delta > 0 else (str(delta) if delta < 0 else "=")
            print(f"  {judge:<30} {hb:>6}/{tot:<3} {hc:>6}/{tot:<3} {arrow:>8}")
        print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
