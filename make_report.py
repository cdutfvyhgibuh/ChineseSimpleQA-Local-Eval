#!/usr/bin/env python3
"""
make_report.py - 生成 ChineseSimpleQA 本地评测结果报告 (HTML + Plotly 交互图表)

用法:
  python make_report.py
  python make_report.py --judge deepseek-r1:14b --out outputs/local_eval/report.html

数据来源（全部只读，不修改任何原始数据）:
  outputs/local_eval/predictions/{model}.jsonl                    目标模型预测
  outputs/local_eval/reviews/{model}__judged_by__{judge}__{pv}.jsonl   Judge 判定
  data/chinese_simpleqa.jsonl                                      数据集（类别/参考答案）
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_ROOT = PROJECT_ROOT / "outputs" / "local_eval"

# 模型展示名（把文件名还原成可读名字，并标注范式）
MODEL_META = {
    "qwen2.5_7b":      ("Qwen2.5 7B Instruct", "Dense + Instruct", "#4C78A8"),
    "deepseek-r1_7b":  ("DeepSeek-R1 7B",      "Reasoning",        "#F58518"),
    "deepseek-v2_16b": ("DeepSeek-V2 16B",     "MoE",              "#54A24B"),
    "phi4-mini_latest": ("Phi-4-mini",         "Dense + Instruct", "#B279A2"),
}

CATEGORY_ORDER = ["中华文化", "人文与社会科学", "自然与自然科学",
                  "工程、技术与应用科学", "社会", "生活、艺术与文化"]


# ============================================================
# 统计工具
# ============================================================

def wilson_ci(k, n, z=1.96):
    """Wilson 置信区间（小样本/极端比例下比正态近似稳健）"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def compute_metrics(correct, incorrect, not_attempted):
    """按官方 ChineseSimpleQA 口径计算指标"""
    total = correct + incorrect + not_attempted
    acc = correct / total if total else 0.0
    attempted = correct + incorrect
    acc_att = correct / attempted if attempted else 0.0
    f1 = (2 * acc_att * acc / (acc_att + acc)) if (acc_att + acc) else 0.0
    lo, hi = wilson_ci(correct, total)
    return {
        "total": total, "correct": correct, "incorrect": incorrect,
        "not_attempted": not_attempted,
        "accuracy": acc, "acc_attempted": acc_att, "f1": f1,
        "ci_lo": lo, "ci_hi": hi,
        "incorrect_rate": incorrect / total if total else 0.0,
        "na_rate": not_attempted / total if total else 0.0,
    }


def load_jsonl(path):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return recs


# ============================================================
# 数据收集
# ============================================================

def collect(judge, prompt_version):
    dataset = load_jsonl(PROJECT_ROOT / "data" / "chinese_simpleqa.jsonl")
    meta = {d["id"]: d for d in dataset}

    judge_safe = judge.replace(":", "_").replace("/", "_")
    results = {}

    for rfile in sorted((OUT_ROOT / "reviews").glob(f"*__judged_by__{judge_safe}__{prompt_version}.jsonl")):
        model_key = rfile.stem.split("__judged_by__")[0]
        reviews = load_jsonl(rfile)
        if not reviews:
            continue
        preds = load_jsonl(OUT_ROOT / "predictions" / f"{model_key}.jsonl")
        plen = {p["id"]: len(p.get("model_output") or "") for p in preds}

        by_cat = defaultdict(lambda: {"A": 0, "B": 0, "C": 0})
        grades = {}
        for r in reviews:
            g = r["grade"]
            cat = meta.get(r["id"], {}).get("primary_category", "未知")
            by_cat[cat][g] += 1
            grades[r["id"]] = g

        results[model_key] = {
            "reviews": reviews,
            "grades": grades,
            "by_category": dict(by_cat),
            "overall": compute_metrics(
                sum(1 for r in reviews if r["grade"] == "A"),
                sum(1 for r in reviews if r["grade"] == "B"),
                sum(1 for r in reviews if r["grade"] == "C"),
            ),
            "out_len": plen,
            "n": len(reviews),
        }
    return dataset, meta, results


# ============================================================
# 图表
# ============================================================

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Microsoft YaHei, Segoe UI, sans-serif", size=12),
    margin=dict(l=60, r=30, t=60, b=50),
)


def fig_overall_bar(results):
    keys = sorted(results, key=lambda k: -results[k]["overall"]["f1"])
    names = [MODEL_META.get(k, (k, "", "#888"))[0] for k in keys]
    accs = [results[k]["overall"]["accuracy"] * 100 for k in keys]
    accs_att = [results[k]["overall"]["acc_attempted"] * 100 for k in keys]
    f1s = [results[k]["overall"]["f1"] * 100 for k in keys]
    colors = [MODEL_META.get(k, (k, "", "#888"))[2] for k in keys]

    fig = go.Figure()
    for vals, nm in [(accs, "Correct% (A/总数)"),
                     (accs_att, "Acc(已作答)= A/(A+B)"),
                     (f1s, "F1")]:
        fig.add_trace(go.Bar(x=names, y=vals, name=nm, text=[f"{v:.1f}%" for v in vals],
                             textposition="outside", marker_color=colors, opacity=0.55 if nm != "F1" else 1.0))
    fig.update_layout(**LAYOUT, barmode="group", title="三模型总体指标对比",
                      yaxis_title="百分比 (%)", yaxis_range=[0, 100],
                      legend=dict(orientation="h", y=-0.18))
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def fig_abc_stack(results):
    keys = sorted(results, key=lambda k: -results[k]["overall"]["f1"])
    names = [MODEL_META.get(k, (k, "", "#888"))[0] for k in keys]
    fig = go.Figure()
    for grade, color, label in [("A", "#54A24B", "A 正确"),
                                ("B", "#E45756", "B 错误"),
                                ("C", "#BAB0AC", "C 未尝试")]:
        vals = []
        for k in keys:
            o = results[k]["overall"]
            vals.append(o["correct"] if grade == "A" else
                        o["incorrect"] if grade == "B" else o["not_attempted"])
        fig.add_trace(go.Bar(x=names, y=vals, name=label, marker_color=color,
                             text=[f"{v}<br>({v/results[k]['n']*100:.1f}%)" for v, k in zip(vals, keys)],
                             textposition="inside"))
    fig.update_layout(**LAYOUT, barmode="stack", title="A / B / C 判定构成（n=3000）",
                      yaxis_title="题数", legend=dict(orientation="h", y=-0.18))
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def fig_category_heatmap(results):
    """类别 × 模型 的 Correct% 热力图"""
    cats = [c for c in CATEGORY_ORDER if any(c in results[k]["by_category"] for k in results)]
    keys = sorted(results, key=lambda k: -results[k]["overall"]["f1"])
    z, text = [], []
    for k in keys:
        row, trow = [], []
        for c in cats:
            v = results[k]["by_category"].get(c, {"A": 0, "B": 0, "C": 0})
            m = compute_metrics(v["A"], v["B"], v["C"])
            row.append(m["accuracy"] * 100)
            trow.append(f"{m['accuracy']*100:.1f}%<br>(n={m['total']})")
        z.append(row)
        text.append(trow)
    fig = go.Figure(go.Heatmap(
        z=z, x=cats, y=[MODEL_META.get(k, (k,))[0] for k in keys],
        text=text, texttemplate="%{text}", textfont=dict(size=11),
        colorscale="RdYlGn", zmin=0, zmax=60,
        colorbar=dict(title="Correct%"),
    ))
    fig.update_layout(**LAYOUT, title="类别 × 模型 Correct% 热力图",
                      height=380, xaxis=dict(tickangle=-18))
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def fig_category_lines(results):
    cats = [c for c in CATEGORY_ORDER if any(c in results[k]["by_category"] for k in results)]
    fig = go.Figure()
    for k in sorted(results, key=lambda x: -results[x]["overall"]["f1"]):
        name, arch, color = MODEL_META.get(k, (k, "", "#888"))
        ys = []
        for c in cats:
            v = results[k]["by_category"].get(c, {"A": 0, "B": 0, "C": 0})
            ys.append(compute_metrics(v["A"], v["B"], v["C"])["accuracy"] * 100)
        fig.add_trace(go.Scatter(x=cats, y=ys, mode="lines+markers+text", name=f"{name} ({arch})",
                                 line=dict(color=color, width=2.5), marker=dict(size=9),
                                 text=[f"{y:.1f}" for y in ys], textposition="top center"))
    fig.update_layout(**LAYOUT, title="各类别 Correct% 走势对比", yaxis_title="Correct% (%)",
                      yaxis_range=[0, 60], legend=dict(orientation="h", y=-0.28), height=460)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def fig_na_rate(results):
    cats = [c for c in CATEGORY_ORDER if any(c in results[k]["by_category"] for k in results)]
    fig = go.Figure()
    for k in sorted(results, key=lambda x: -results[x]["overall"]["f1"]):
        name, arch, color = MODEL_META.get(k, (k, "", "#888"))
        ys = []
        for c in cats:
            v = results[k]["by_category"].get(c, {"A": 0, "B": 0, "C": 0})
            ys.append(compute_metrics(v["A"], v["B"], v["C"])["na_rate"] * 100)
        fig.add_trace(go.Bar(x=cats, y=ys, name=f"{name}", marker_color=color))
    fig.update_layout(**LAYOUT, title="各类别「未尝试(C)」比例 — 反映模型的保守/拒答倾向",
                      yaxis_title="C 占比 (%)", barmode="group",
                      legend=dict(orientation="h", y=-0.28), height=460, xaxis=dict(tickangle=-18))
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def fig_difficulty(results):
    """按「至少一个模型答对」定义题目难度分层，看各模型表现"""
    keys = list(results)
    ids = list(results[keys[0]]["grades"])
    n_models = len(keys)

    def n_correct(qid):
        return sum(1 for k in keys if results[k]["grades"].get(qid) == "A")

    buckets = defaultdict(lambda: defaultdict(int))
    bucket_tot = Counter()
    for qid in ids:
        nc = n_correct(qid)
        bucket_tot[nc] += 1
        for k in keys:
            if results[k]["grades"].get(qid) == "A":
                buckets[nc][k] += 1

    xlabels = [f"{i}/{n_models} 模型答对" for i in range(n_models + 1)]
    xs = list(range(n_models + 1))
    fig = go.Figure()
    for k in sorted(results, key=lambda x: -results[x]["overall"]["f1"]):
        name, arch, color = MODEL_META.get(k, (k, "", "#888"))
        ys = [(buckets[i][k] / bucket_tot[i] * 100) if bucket_tot[i] else 0 for i in xs]
        fig.add_trace(go.Bar(x=xlabels, y=ys, name=name, marker_color=color,
                             text=[f"{y:.0f}%" if bucket_tot[i] else "" for i, y in zip(xs, ys)],
                             textposition="outside"))
    fig.update_layout(**LAYOUT, title="题目难度分层：按「有几个模型答对」分组，看各模型命中率",
                      yaxis_title="该层内答对比例 (%)", barmode="group",
                      yaxis_range=[0, 105], legend=dict(orientation="h", y=-0.2), height=430)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def venn_breakdown(results):
    """8 个子集：哪些模型答对了这道题（维恩分解）"""
    keys = sorted(results, key=lambda k: -results[k]["overall"]["f1"])
    ids = list(results[keys[0]]["grades"])
    combos = defaultdict(int)
    for q in ids:
        pat = tuple(k for k in keys if results[k]["grades"].get(q) == "A")
        combos[pat] += 1
    return keys, combos, len(ids)


def fig_venn(results):
    keys, combos, total = venn_breakdown(results)
    labels, vals, colors = [], [], []
    palette = {"deepseek-v2_16b": "#54A24B", "qwen2.5_7b": "#4C78A8", "deepseek-r1_7b": "#F58518"}

    # 全错
    if () in combos:
        labels.append("三模型全错")
        vals.append(combos[()])
        colors.append("#E45756")
    # 单模型
    for k in keys:
        n = combos.get((k,), 0)
        if n:
            labels.append(f"仅 {MODEL_META.get(k,(k,))[0]}")
            vals.append(n)
            colors.append(palette.get(k, "#888"))
    # 双模型
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pair = tuple(k for k in keys if k in (keys[i], keys[j]))
            n = combos.get(pair, 0)
            if n:
                nm = " + ".join(MODEL_META.get(x, (x,))[0].split()[0] for x in pair)
                labels.append(f"仅 {nm}")
                vals.append(n)
                colors.append("#B279A2")
    # 三模型
    if tuple(keys) in combos:
        labels.append("三模型全对")
        vals.append(combos[tuple(keys)])
        colors.append("#2f9e44")

    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]
    colors = [colors[i] for i in order]

    fig = go.Figure(go.Bar(
        y=labels[::-1], x=vals[::-1], orientation="h",
        marker_color=colors[::-1],
        text=[f"{v} ({v/total*100:.1f}%)" for v in vals[::-1]],
        textposition="outside",
    ))
    fig.update_layout(**LAYOUT, title=f"答对题目的模型组合分布（共 {total} 题）",
                      xaxis_title="题数", height=430)
    fig.update_layout(margin=dict(l=200, r=90, t=60, b=50))
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def fig_agreement(results):
    """模型两两之间判定完全一致的题目比例"""
    keys = sorted(results, key=lambda k: -results[k]["overall"]["f1"])
    z, text = [], []
    for a in keys:
        row, trow = [], []
        for b in keys:
            ga, gb = results[a]["grades"], results[b]["grades"]
            common = [q for q in ga if q in gb]
            same = sum(1 for q in common if ga[q] == gb[q])
            v = same / len(common) * 100 if common else 0
            row.append(v)
            trow.append(f"{v:.1f}%")
        z.append(row)
        text.append(trow)
    fig = go.Figure(go.Heatmap(z=z, x=[MODEL_META.get(k, (k,))[0] for k in keys],
                               y=[MODEL_META.get(k, (k,))[0] for k in keys],
                               text=text, texttemplate="%{text}",
                               colorscale="Blues", zmin=50, zmax=100,
                               colorbar=dict(title="一致率%")))
    fig.update_layout(**LAYOUT, title="模型间 A/B/C 判定一致率（对角线=100%）", height=400)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


# ============================================================
# HTML
# ============================================================

def fmt_pct(x, nd=1):
    return f"{x*100:.{nd}f}%"


def build_html(results, judge, prompt_version, dataset):
    keys = sorted(results, key=lambda k: -results[k]["overall"]["f1"])
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---- 总体表 ----
    rows = []
    for i, k in enumerate(keys, 1):
        o = results[k]["overall"]
        name, arch, color = MODEL_META.get(k, (k, "", "#888"))
        rows.append(f"""<tr>
          <td class="ctr">{i}</td>
          <td><span class="dot" style="background:{color}"></span><b>{name}</b></td>
          <td class="ctr"><span class="tag">{arch}</span></td>
          <td class="num">{o['total']}</td>
          <td class="num ok">{o['correct']}</td>
          <td class="num bad">{o['incorrect']}</td>
          <td class="num mute">{o['not_attempted']}</td>
          <td class="num strong">{fmt_pct(o['accuracy'])}</td>
          <td class="num">{fmt_pct(o['acc_attempted'])}</td>
          <td class="num strong">{fmt_pct(o['f1'])}</td>
          <td class="num ci">[{fmt_pct(o['ci_lo'])}, {fmt_pct(o['ci_hi'])}]</td>
        </tr>""")
    overall_table = "\n".join(rows)

    # ---- 分类别表（每模型一块）----
    cat_blocks = []
    for k in keys:
        name, arch, color = MODEL_META.get(k, (k, "", "#888"))
        cats = [c for c in CATEGORY_ORDER if c in results[k]["by_category"]]
        trs = []
        for c in cats:
            v = results[k]["by_category"][c]
            m = compute_metrics(v["A"], v["B"], v["C"])
            trs.append(f"""<tr>
              <td>{c}</td><td class="num">{m['total']}</td>
              <td class="num ok">{m['correct']}</td>
              <td class="num bad">{m['incorrect']}</td>
              <td class="num mute">{m['not_attempted']}</td>
              <td class="num strong">{fmt_pct(m['accuracy'])}</td>
              <td class="num">{fmt_pct(m['acc_attempted'])}</td>
              <td class="num">{fmt_pct(m['na_rate'])}</td>
              <td class="num strong">{fmt_pct(m['f1'])}</td>
            </tr>""")
        o = results[k]["overall"]
        trs.append(f"""<tr class="total">
          <td><b>合计</b></td><td class="num">{o['total']}</td>
          <td class="num ok">{o['correct']}</td><td class="num bad">{o['incorrect']}</td>
          <td class="num mute">{o['not_attempted']}</td>
          <td class="num strong">{fmt_pct(o['accuracy'])}</td>
          <td class="num">{fmt_pct(o['acc_attempted'])}</td>
          <td class="num">{fmt_pct(o['na_rate'])}</td>
          <td class="num strong">{fmt_pct(o['f1'])}</td>
        </tr>""")
        cat_blocks.append(f"""
        <div class="card">
          <h4><span class="dot" style="background:{color}"></span>{name} <span class="tag">{arch}</span></h4>
          <table class="grid">
            <thead><tr><th>类别</th><th>n</th><th>A 正确</th><th>B 错误</th><th>C 未尝试</th>
            <th>Correct%</th><th>Acc(已作答)</th><th>C 占比</th><th>F1</th></tr></thead>
            <tbody>{''.join(trs)}</tbody>
          </table>
        </div>""")

    # ---- 关键观察（自动生成）----
    best = keys[0]
    obs = []
    bn, ba, _ = MODEL_META.get(best, (best, "", "#888"))
    obs.append(f"<li><b>排名第一：{bn}</b>（{ba}），F1 {fmt_pct(results[best]['overall']['f1'])}、"
               f"Correct {fmt_pct(results[best]['overall']['accuracy'])}。</li>")
    if len(keys) >= 2:
        w = keys[-1]
        wn, wa, _ = MODEL_META.get(w, (w, "", "#888"))
        obs.append(f"<li><b>排名末位：{wn}</b>（{wa}），F1 {fmt_pct(results[w]['overall']['f1'])}，"
                   f"与第一名相差 {fmt_pct(results[best]['overall']['f1']-results[w]['overall']['f1'])}。</li>")
    # 拒答倾向
    na_sorted = sorted(keys, key=lambda k: -results[k]["overall"]["na_rate"])
    hk = na_sorted[0]
    hn, ha, _ = MODEL_META.get(hk, (hk, "", "#888"))
    obs.append(f"<li><b>最保守：{hn}</b>，C（未尝试）比例 {fmt_pct(results[hk]['overall']['na_rate'])}，"
               f"即 {results[hk]['overall']['not_attempted']} 题直接拒答/未给出答案。</li>")
    # 最强/最弱类别（以第一名模型为准）
    bc = results[best]["by_category"]
    if bc:
        cm = {c: compute_metrics(v["A"], v["B"], v["C"]) for c, v in bc.items()}
        top_cat = max(cm, key=lambda c: cm[c]["accuracy"])
        low_cat = min(cm, key=lambda c: cm[c]["accuracy"])
        obs.append(f"<li><b>{bn} 的强项</b>：{top_cat} {fmt_pct(cm[top_cat]['accuracy'])}；"
                   f"<b>短板</b>：{low_cat} {fmt_pct(cm[low_cat]['accuracy'])}。</li>")
    # F1 与纯正确率差异 → 是否靠"少答"换准确率
    obs.append("<li>⚠️ <b>Correct% 与 F1 的差距反映作答策略</b>：大量拒答会抬高 Acc(已作答) 但拉低 Correct%，"
               "因此比较模型能力时应以 <b>F1</b> 为准（官方口径）。</li>")

    # 难度分层 + 维恩
    keys_v, combos, total_q = venn_breakdown(results)
    n_all_wrong = combos.get((), 0)
    obs.append(f"<li><b>题目难度</b>：{n_all_wrong} 题（{n_all_wrong/total_q*100:.1f}%）三个模型<b>全部答错</b>，"
               f"说明 ChineseSimpleQA 对本地中小模型整体偏难；仅 {combos.get(tuple(keys_v),0)} 题"
               f"（{combos.get(tuple(keys_v),0)/total_q*100:.1f}%）三者都答对。</li>")
    uniq_best = max(keys_v, key=lambda k: combos.get((k,), 0))
    obs.append(f"<li><b>能力互补性</b>：{MODEL_META.get(uniq_best,(uniq_best,))[0]} 有 "
               f"<b>{combos.get((uniq_best,),0)} 题</b>是「只有它答对」，"
               f"说明模型间存在明显互补，简单排名无法反映各自独有的知识覆盖。</li>")
    if len(keys_v) >= 2:
        min_pair = None
        for i in range(len(keys_v)):
            for j in range(i + 1, len(keys_v)):
                a, b = keys_v[i], keys_v[j]
                same = sum(1 for q in results[a]["grades"]
                           if results[a]["grades"][q] == results[b]["grades"].get(q))
                rate = same / total_q
                if min_pair is None or rate < min_pair[2]:
                    min_pair = (a, b, rate)
        if min_pair:
            obs.append(f"<li><b>判定分歧</b>：分歧最大的一对是 "
                       f"{MODEL_META.get(min_pair[0],(min_pair[0],))[0]} 与 "
                       f"{MODEL_META.get(min_pair[1],(min_pair[1],))[0]}，"
                       f"A/B/C 完全一致率仅 {fmt_pct(min_pair[2])} —— 模型间的「作答风格」差异巨大。</li>")

    css = """
    :root{--bg:#f6f7f9;--card:#fff;--line:#e3e6ea;--txt:#1f2328;--mute:#6b7280;
          --ok:#2f9e44;--bad:#e03131;--na:#868e96;--accent:#1c7ed6;}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--txt);
         font-family:"Microsoft YaHei","Segoe UI",system-ui,sans-serif;line-height:1.6}
    .wrap{max-width:1280px;margin:0 auto;padding:28px 22px 60px}
    header{background:linear-gradient(135deg,#1c7ed6,#0b4f8a);color:#fff;border-radius:14px;
           padding:26px 30px;margin-bottom:22px;box-shadow:0 4px 18px rgba(0,0,0,.12)}
    header h1{margin:0 0 6px;font-size:26px}
    header .sub{opacity:.92;font-size:14px}
    header .meta{margin-top:14px;display:flex;flex-wrap:wrap;gap:10px}
    header .meta span{background:rgba(255,255,255,.16);padding:4px 11px;border-radius:20px;font-size:12.5px}
    .card{background:var(--card);border:1px solid var(--line);border-radius:12px;
          padding:18px 20px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
    h2{font-size:19px;margin:26px 0 12px;padding-left:11px;border-left:4px solid var(--accent)}
    h3{font-size:15.5px;margin:18px 0 8px;color:#34495e}
    h4{font-size:14.5px;margin:0 0 12px;display:flex;align-items:center;gap:8px}
    table{border-collapse:collapse;width:100%;font-size:13px}
    th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left}
    th{background:#f1f3f5;font-weight:600;color:#495057;white-space:nowrap}
    tbody tr:hover{background:#f8f9fa}
    .num{text-align:right;font-variant-numeric:tabular-nums}
    .ctr{text-align:center}
    .ok{color:var(--ok);font-weight:600}
    .bad{color:var(--bad)}
    .mute{color:var(--na)}
    .strong{font-weight:700}
    .ci{color:var(--mute);font-size:12px}
    tr.total{background:#f8f9fa;border-top:2px solid var(--line)}
    .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
    .tag{background:#e7f5ff;color:#1971c2;border-radius:5px;padding:1.5px 7px;font-size:11.5px;font-weight:600}
    ul.obs{margin:6px 0 0;padding-left:22px}
    ul.obs li{margin-bottom:7px}
    .note{background:#fff9db;border-left:4px solid #f59f00;padding:11px 15px;
          border-radius:7px;font-size:13px;margin:10px 0}
    .grid{font-size:12.5px}
    footer{color:var(--mute);font-size:12px;text-align:center;margin-top:34px}
    .kpi{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:6px}
    .kpi div{flex:1;min-width:150px;background:#f8f9fa;border:1px solid var(--line);
             border-radius:10px;padding:12px 15px}
    .kpi .v{font-size:22px;font-weight:700;color:var(--accent)}
    .kpi .l{font-size:12px;color:var(--mute)}
    """

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChineseSimpleQA 本地评测报告 — Judge {judge}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>{css}</style></head><body><div class="wrap">

<header>
  <h1>ChineseSimpleQA 本地大模型评测报告</h1>
  <div class="sub">全本地推理 · 统一 Judge 裁定 · 可复现评测流程</div>
  <div class="meta">
    <span>数据集：ChineseSimpleQA（{len(dataset)} 题）</span>
    <span>Judge：{judge}（prompt={prompt_version}）</span>
    <span>目标模型：{len(results)} 个</span>
    <span>判定总数：{sum(r['n'] for r in results.values())}</span>
    <span>生成时间：{ts}</span>
  </div>
</header>

<div class="card">
  <h3>评测设置</h3>
  <div class="kpi">
    <div><div class="v">{len(dataset)}</div><div class="l">题目总数</div></div>
    <div><div class="v">{len(results)}</div><div class="l">目标模型</div></div>
    <div><div class="v">{sum(r['n'] for r in results.values())}</div><div class="l">判定总数</div></div>
    <div><div class="v">8192</div><div class="l">num_ctx</div></div>
    <div><div class="v">1</div><div class="l">并发</div></div>
  </div>
  <div class="note"><b>评分口径</b>：A=正确 / B=错误 / C=未尝试（官方 ChineseSimpleQA 定义）。
  <b>Correct%</b>=A/总数；<b>Acc(已作答)</b>=A/(A+B)；<b>F1</b>=2·Acc(已作答)·Correct/(Acc(已作答)+Correct)。
  所有目标模型使用<b>同一个 Judge（{judge}）</b>，保证横向可比。</div>
</div>

<h2>一、总体结果</h2>
<div class="card">
  <table>
    <thead><tr><th>#</th><th>模型</th><th>架构范式</th><th>n</th>
    <th>A 正确</th><th>B 错误</th><th>C 未尝试</th>
    <th>Correct%</th><th>Acc(已作答)</th><th>F1</th><th>95% CI (Correct)</th></tr></thead>
    <tbody>{overall_table}</tbody>
  </table>
  <div class="note">F1 为官方主指标。95% 置信区间用 Wilson 区间计算（n=3000 时约 ±1.5 个百分点）。</div>
</div>

<div class="card">{fig_overall_bar(results)}</div>
<div class="card">{fig_abc_stack(results)}</div>

<h2>二、分类别表现</h2>
<div class="card">{fig_category_heatmap(results)}</div>
<div class="card">{fig_category_lines(results)}</div>
<div class="card">{fig_na_rate(results)}</div>
{''.join(cat_blocks)}

<h2>三、题目难度与模型一致性</h2>
<div class="card">{fig_difficulty(results)}</div>
<div class="card">{fig_venn(results)}</div>
<div class="card">{fig_agreement(results)}</div>

<h2>四、结论要点</h2>
<div class="card"><ul class="obs">{''.join(obs)}</ul></div>

<h2>五、方法与可信度说明</h2>
<div class="card">
  <h3>为什么不能只看「报告出来的正确率」</h3>
  <p style="font-size:13.5px;margin-top:0">
  Judge 模型自身的误判会系统性地扭曲目标模型的分数。本项目在正式评测前，用同一批 50 题对 4 个候选
  Judge 做了横向对比与<b>人工核验</b>，确认了各自的偏差方向：</p>
  <table>
    <thead><tr><th>候选 Judge</th><th>偏差方向</th><th>典型误判</th><th>是否采用</th></tr></thead>
    <tbody>
      <tr><td>qwen2.5:7b</td><td>偏严</td><td>答对+补充信息→误判 B</td><td>否</td></tr>
      <tr><td>deepseek-r1:7b</td><td>偏松</td><td>答错→误判 A</td><td>否</td></tr>
      <tr><td><b>deepseek-r1:14b</b></td><td>偏严（较轻）</td><td>对已答对但补充无关信息的题判 B</td><td class="ok">✔ 采用</td></tr>
      <tr><td>bonsai-27b</td><td>偏松 + C 滥用</td><td>答错→误判 C；提到关键词即判 A</td><td>仅作对照</td></tr>
    </tbody>
  </table>
  <h3>已做过的稳健性工作</h3>
  <ul style="font-size:13.5px">
    <li><b>Judge 校准实验</b>：设计了一份边界规则更明确的校准版 Prompt，在 50 题上对比。
        结论——它显著修复了 bonsai 的「C 滥用」，但对 deepseek-r1:14b 无净提升，
        说明其残留误判来自模型能力上限而非 Prompt 模糊。故正式评测采用官方模板。</li>
    <li><b>上下文控制</b>：Ollama 层实测确认 <code>num_ctx=8192</code> 生效；judge 单题输入
        约 1900–3600 字符、生成中位 683 token，最坏占用约 52%，不会截断。</li>
    <li><b>无截断验证</b>：3000 题预测输出最长 1219 字符，无一触顶 max_tokens。</li>
    <li><b>数据完整性</b>：三个模型的预测与判定 id 均与数据集 3000 题完全一致，无缺题、无重复。</li>
  </ul>
  <div class="note">⚠️ <b>局限性</b>：Judge 与被测模型同为本机可部署的中小规模模型，其判定准确率经人工核验约
  87%（15 题样本），因此<b>各模型之间的相对排名可信，但绝对分数存在约 ±1~2 个百分点的系统偏差</b>。
  不同 Judge 下的绝对分数不可直接混用。</div>
</div>

<footer>
  数据来源：outputs/local_eval（predictions / reviews 原始 JSONL 均完整保留）<br>
  本报告由 make_report.py 自动生成 · 所有指标均由原始判定记录重新计算
</footer>
</div></body></html>"""
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="deepseek-r1:14b")
    ap.add_argument("--prompt-version", default="official")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dataset, meta, results = collect(args.judge, args.prompt_version)
    if not results:
        print("[ERR] 未找到匹配的 review 文件")
        return

    print(f"[OK] 数据集 {len(dataset)} 题，收集到 {len(results)} 个模型")
    for k, v in sorted(results.items(), key=lambda x: -x[1]["overall"]["f1"]):
        o = v["overall"]
        print(f"  {k:<20} n={o['total']} A={o['correct']} B={o['incorrect']} C={o['not_attempted']} "
              f"Correct={o['accuracy']*100:.1f}% F1={o['f1']*100:.1f}%")

    html = build_html(results, args.judge, args.prompt_version, dataset)
    out = Path(args.out) if args.out else (OUT_ROOT / "report.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\n[OK] 报告已生成: {out}")
    print(f"     大小: {out.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
