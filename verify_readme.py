#!/usr/bin/env python3
"""
verify_readme.py - 核对 README.md 中引用的每个数字是否与原始数据一致

用法:
  python verify_readme.py            # 核对发布目录内的 README.md
  python verify_readme.py --strict   # 任何不一致则退出码非 0

设计原则：所有断言都从 results/ 下的原始 JSONL 现场重算，
不读任何中间缓存，确保"文章里的数字 = 原始数据算出来的数字"。
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
README = os.path.join(HERE, "README.md")

CATEGORY_ORDER = ["中华文化", "人文与社会科学", "自然与自然科学",
                  "工程、技术与应用科学", "社会", "生活、艺术与文化"]

# 文章正文中需要被核对的数字（期望值 -> 描述），由 recompute() 填充
CHECKS = []


def add(desc, expected, actual, tol=0.005):
    """tol 用于百分比比较（0.005 = 0.005 个百分点）"""
    ok = abs(expected - actual) <= tol
    CHECKS.append((ok, desc, expected, actual))


def metrics(grades):
    c = Counter(grades.values())
    A, B, C = c["A"], c["B"], c["C"]
    n = len(grades)
    acc = A / n
    att = A / (A + B) if (A + B) else 0.0
    f1 = 2 * att * acc / (att + acc) if (att + acc) else 0.0
    return dict(n=n, A=A, B=B, C=C, acc=acc, att=att, f1=f1)


def recompute():
    # ---- 数据集 ----
    ds = {}
    with open(os.path.join(HERE, "data", "chinese_simpleqa.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                ds[o["id"]] = o
    add("数据集题目数", 3000, len(ds), 0)

    # ---- 判定结果 ----
    G = {}
    for fp in sorted(glob.glob(os.path.join(RESULTS, "reviews", "*.jsonl"))):
        key = os.path.basename(fp).split("__judged_by__")[0]
        g = {}
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    g[o["id"]] = o["grade"]
        G[key] = g

    add("模型数量", 3, len(G), 0)
    for k, g in G.items():
        add(f"{k} 判定总数", 3000, len(g), 0)
    total = sum(len(g) for g in G.values())
    add("判定总条数", 9000, total, 0)

    # ---- 总体指标（文章 8.1 表格）----
    EXPECT = {
        "deepseek-v2_16b":  dict(A=1110, B=1464, C=426, acc=37.00, att=43.12, f1=39.83),
        "qwen2.5_7b":       dict(A=980,  B=1796, C=224, acc=32.67, att=35.30, f1=33.93),
        "deepseek-r1_7b":   dict(A=301,  B=2032, C=667, acc=10.03, att=12.90, f1=11.29),
    }
    M = {}
    for k, g in G.items():
        m = metrics(g)
        M[k] = m
        e = EXPECT[k]
        add(f"{k} A", e["A"], m["A"], 0)
        add(f"{k} B", e["B"], m["B"], 0)
        add(f"{k} C", e["C"], m["C"], 0)
        add(f"{k} Correct%", e["acc"], m["acc"] * 100)
        add(f"{k} Acc(已作答)%", e["att"], m["att"] * 100)
        add(f"{k} F1%", e["f1"], m["f1"] * 100)

    # 排序：文章称 V2 > Qwen > R1
    order = sorted(G, key=lambda x: -M[x]["f1"])
    add("F1 排名第1=V2_16b", 0,
        0 if order[0] == "deepseek-v2_16b" else 1, 0)
    add("F1 排名第2=Qwen_7b", 0,
        0 if order[1] == "qwen2.5_7b" else 1, 0)
    add("F1 排名第3=R1_7b", 0,
        0 if order[2] == "deepseek-r1_7b" else 1, 0)

    # ---- 分类别（文章 8.2 表格）----
    CAT_EXPECT = {
        "中华文化":            (326, 44.79, 28.83, 5.21),
        "人文与社会科学":       (609, 41.54, 38.26, 8.21),
        "自然与自然科学":       (530, 33.40, 36.04, 16.42),
        "工程、技术与应用科学":  (481, 40.12, 38.25, 16.84),
        "社会":                (453, 35.10, 34.44, 9.49),
        "生活、艺术与文化":     (601, 30.28, 20.30, 3.83),
    }
    for cat, (n_exp, v2, qw, r1) in CAT_EXPECT.items():
        ids = [q for q in ds if ds[q]["primary_category"] == cat]
        add(f"[{cat}] n", n_exp, len(ids), 0)
        for key, exp in [("deepseek-v2_16b", v2), ("qwen2.5_7b", qw), ("deepseek-r1_7b", r1)]:
            a = sum(1 for q in ids if G[key].get(q) == "A")
            add(f"[{cat}] {key} Correct%", exp, a / len(ids) * 100)

    # ---- 难度分层（文章 8.3 ④）----
    keys = list(G)
    ids_all = list(ds)
    nc = Counter(sum(1 for k in keys if G[k].get(q) == "A") for q in ids_all)
    for i, exp_cnt, exp_pct in [(0, 1526, 50.87), (1, 724, 24.13),
                                (2, 583, 19.43), (3, 167, 5.57)]:
        add(f"难度分层 {i}/3 题数", exp_cnt, nc[i], 0)
        add(f"难度分层 {i}/3 占比", exp_pct, nc[i] / len(ids_all) * 100)

    # ---- 独有答对（文章 8.3 ③）----
    for k, exp in [("deepseek-v2_16b", 409), ("qwen2.5_7b", 268), ("deepseek-r1_7b", 47)]:
        u = sum(1 for q in ids_all
                if G[k].get(q) == "A" and all(G[o].get(q) != "A" for o in keys if o != k))
        add(f"{k} 独有答对", exp, u, 0)

    # ---- 一致率矩阵（文章 8.3 ③）----
    AGREE = {
        ("deepseek-r1_7b", "deepseek-v2_16b"): 47.53,
        ("deepseek-r1_7b", "qwen2.5_7b"): 52.90,
        ("deepseek-v2_16b", "qwen2.5_7b"): 61.97,
    }
    for (a, b), exp in AGREE.items():
        same = sum(1 for q in ids_all if G[a].get(q) == G[b].get(q))
        add(f"一致率 {a[:14]} vs {b[:14]}", exp, same / len(ids_all) * 100)

    # ---- 校准实验（文章第 7 节）----
    CAL_EXPECT = {
        "calibrated__deepseek-r1_14b.jsonl":            dict(A=16, B=31, C=3),
        "calibrated__MichelRosselli_bonsai-27b.jsonl":  dict(A=18, B=27, C=5),
    }
    for fn, e in CAL_EXPECT.items():
        fp = os.path.join(RESULTS, "calibration", fn)
        c = Counter()
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c[json.loads(line)["grade"]] += 1
        for g_ in "ABC":
            add(f"校准 {fn[:36]} {g_}", e[g_], c[g_], 0)

    # ---- 截断检查（文章第 9 节）----
    maxlen = 0
    for fp in glob.glob(os.path.join(RESULTS, "predictions", "*.jsonl")):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    maxlen = max(maxlen, len(json.loads(line).get("model_output") or ""))
    add("预测输出最长字符数", 1219, maxlen, 0)

    # ---- R1-7B 拒答数（文章 8.3 ②）----
    add("R1_7b 拒答(C)数", 667, M["deepseek-r1_7b"]["C"], 0)
    add("Qwen_7b 拒答(C)数", 224, M["qwen2.5_7b"]["C"], 0)
    add("R1_7b C 占比%", 22.23, M["deepseek-r1_7b"]["C"] / 3000 * 100, 0.02)
    add("Qwen_7b C 占比%", 7.47, M["qwen2.5_7b"]["C"] / 3000 * 100, 0.02)

    return M, G


def check_readme_numbers():
    """确认文章里出现的关键数字字符串确实存在于 README（防止文章与断言脱节）"""
    if not os.path.exists(README):
        print(f"[WARN] 未找到 {README}，跳过文本存在性检查")
        return
    txt = open(README, encoding="utf-8").read()
    must_have = [
        "LivingFutureLab/ChineseSimpleQA", "39.83", "33.93", "11.29",
        "50.87", "47.53", "61.97", "409", "1526",
        "1755", "金塔", "第三十二代",
    ]
    missing = [s for s in must_have if s not in txt]
    if missing:
        print(f"[WARN] README 中缺少以下关键内容: {missing}")
    else:
        print("[OK] README 关键内容齐全（含上游引用、核心指标、事实核实）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("  README 数字核对 — 全部从 results/ 原始 JSONL 现场重算")
    print("=" * 74)

    recompute()

    fails = [c for c in CHECKS if not c[0]]
    for ok, desc, exp, act in CHECKS:
        if not ok:
            print(f"  ✗ {desc}")
            print(f"      文章写: {exp}    实际算: {act}")

    print()
    print(f"  核对项: {len(CHECKS)}   通过: {len(CHECKS)-len(fails)}   不一致: {len(fails)}")
    print()
    check_readme_numbers()
    print("=" * 74)

    if fails and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
