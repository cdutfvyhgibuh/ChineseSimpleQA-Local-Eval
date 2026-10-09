#!/usr/bin/env python3
"""
run_judge_calibration.py - Judge 校准实验
用校准版 Prompt 重跑 50 题，对比 R1-14B 和 Bonsai-27B 的判定质量。

用法:
  python run_judge_calibration.py --judge deepseek-r1:14b
  python run_judge_calibration.py --judge MichelRosselli/bonsai-27b
  python run_judge_calibration.py --judge deepseek-r1:14b --judge MichelRosselli/bonsai-27b
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTIONS_FILE = PROJECT_ROOT / "outputs" / "local_eval" / "predictions" / "qwen2.5_7b.jsonl"
CALIBRATION_DIR = PROJECT_ROOT / "outputs" / "local_eval" / "calibration"

# 校准版模板统一定义在 run_local_eval.py，避免两处副本漂移。
# run_local_eval.py 的模块级代码只有常量/函数定义，import 无副作用。
sys.path.insert(0, str(PROJECT_ROOT))
from run_local_eval import (  # noqa: E402
    CALIBRATED_GRADER_TEMPLATE,
    CALIBRATED_JUDGE_SYSTEM_MESSAGE as JUDGE_SYSTEM_MESSAGE,
)

# ============================================================
# 工具函数（复用 run_local_eval.py 的逻辑）
# ============================================================

def sanitize_model_name(model: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]', '_', model)


def parse_grade(judge_raw: str) -> str:
    THINK_OPEN = "<" + "think>"
    THINK_CLOSE = "</" + "think>"
    pattern_closed = re.escape(THINK_OPEN) + r".*?" + re.escape(THINK_CLOSE)
    text = re.sub(pattern_closed, "", judge_raw, flags=re.DOTALL)
    pattern_open = re.escape(THINK_OPEN) + r".*"
    text = re.sub(pattern_open, "", text, flags=re.DOTALL)
    text = text.strip()
    matches = re.findall(r"\b([ABC])\b", text)
    if matches:
        return matches[-1]
    matches_all = re.findall(r"\b([ABC])\b", judge_raw)
    if matches_all:
        return matches_all[-1]
    return "C"


def call_ollama_chat(client, model, messages, temperature=0.0, max_tokens=4096,
                     timeout=300, max_retries=3, retry_backoff=5, num_ctx=8192):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                extra_body={"options": {"num_ctx": num_ctx}},
            )
            content = response.choices[0].message.content
            if content is not None:
                return content.strip()
        except Exception as e:
            print(f"    [ERR] {type(e).__name__}: {e} (attempt {attempt+1}/{max_retries})")
        if attempt < max_retries - 1:
            time.sleep(retry_backoff * (attempt + 1))
    return None


def ollama_unload(base_url, model):
    host = base_url.replace('/v1', '').replace('/v1/', '')
    try:
        requests.post(f"{host}/api/generate",
                      json={"model": model, "prompt": "", "keep_alive": 0}, timeout=10)
        print(f"  [VRAM] 已卸载: {model}")
    except Exception:
        pass


# ============================================================
# 主逻辑
# ============================================================

def run_calibration(judge_model: str, predictions: list[dict], base_url: str, num_ctx: int = 8192):
    """用校准 Prompt 跑一个 Judge 模型"""
    judge_safe = sanitize_model_name(judge_model)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    output_file = CALIBRATION_DIR / f"calibrated__{judge_safe}.jsonl"

    # 断点续跑
    done_ids = set()
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        done_ids.add(obj["id"])
                    except json.JSONDecodeError:
                        continue

    todo = [p for p in predictions if p["id"] not in done_ids and p.get("status") == "ok"]

    print(f"\n{'='*60}")
    print(f"[CALIBRATION] Judge: {judge_model}")
    print(f"  Prompt: calibrated (explicit A/B/C rules)")
    print(f"  已评: {len(done_ids)}, 待评: {len(todo)}")
    print(f"  输出: {output_file}")
    print(f"{'='*60}")

    if not todo:
        print("  全部已完成。")
        return output_file

    client = OpenAI(api_key="ollama", base_url=base_url)
    t0 = time.time()

    for pred in tqdm(todo, desc=f"  {judge_model}", unit="q"):
        grader_prompt = CALIBRATED_GRADER_TEMPLATE.format(
            question=pred["question"],
            target=pred["answer"],
            predicted_answer=pred["model_output"],
        )
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_MESSAGE},
            {"role": "user", "content": grader_prompt},
        ]
        judge_raw = call_ollama_chat(client, judge_model, messages,
                                     temperature=0.0, max_tokens=4096,
                                     timeout=300, num_ctx=num_ctx)

        grade_letter = parse_grade(judge_raw) if judge_raw else "C"

        record = {
            "id": pred["id"],
            "model": pred["model"],
            "judge_model": judge_model,
            "prompt_version": "calibrated_v1",
            "primary_category": pred.get("primary_category", ""),
            "question": pred["question"],
            "answer": pred["answer"],
            "model_output": pred["model_output"],
            "judge_raw": judge_raw if judge_raw else "",
            "grade": grade_letter,
            "timestamp": datetime.now().isoformat(),
        }
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()

    elapsed = time.time() - t0
    print(f"  完成, 耗时={elapsed/60:.1f}min, 平均={elapsed/len(todo):.1f}s/题")
    ollama_unload(base_url, judge_model)
    return output_file


def print_comparison(output_files: list[Path]):
    """对比多个 Judge 的校准结果"""
    print(f"\n{'='*70}")
    print("  CALIBRATION COMPARISON")
    print(f"{'='*70}")

    all_results = {}
    for f in output_files:
        if not f.exists():
            continue
        records = []
        with open(f, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        judge_name = records[0]["judge_model"] if records else f.stem
        all_results[judge_name] = {r["id"]: r["grade"] for r in records}

    if not all_results:
        print("  无结果文件")
        return

    # 统计
    judges = list(all_results.keys())
    print(f"\n  Judges: {judges}")
    print(f"  Questions: {len(next(iter(all_results.values())))}")

    for judge in judges:
        grades = list(all_results[judge].values())
        a_count = grades.count("A")
        b_count = grades.count("B")
        c_count = grades.count("C")
        total = len(grades)
        print(f"\n  [{judge}]")
        print(f"    A={a_count} ({a_count/total*100:.0f}%)  "
              f"B={b_count} ({b_count/total*100:.0f}%)  "
              f"C={c_count} ({c_count/total*100:.0f}%)")

    # 逐题对比（只列不一致的）
    if len(judges) >= 2:
        print(f"\n  {'#':<3} {'Question':<30} ", end="")
        for j in judges:
            short = j.split("/")[-1] if "/" in j else j
            print(f"{short:<16}", end="")
        print()
        print(f"  {'-'*(3+30+16*len(judges))}")

        ids = list(next(iter(all_results.values())).keys())
        # 读取问题文本
        questions = {}
        for f in output_files:
            if f.exists():
                with open(f, 'r', encoding='utf-8') as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                r = json.loads(line)
                                questions[r["id"]] = r["question"][:28]
                            except:
                                pass
                break

        diff_count = 0
        for i, qid in enumerate(ids):
            grades = [all_results[j].get(qid, "?") for j in judges]
            if len(set(grades)) > 1:  # 只打印不一致的
                diff_count += 1
                q_text = questions.get(qid, "?")
                print(f"  {i+1:<3} {q_text:<30} ", end="")
                for g in grades:
                    print(f"{g:<16}", end="")
                print()

        print(f"\n  不一致题数: {diff_count}/{len(ids)}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Judge Calibration Experiment")
    parser.add_argument("--judge", type=str, nargs="+", required=True,
                        help="Judge model(s) to calibrate")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config_local.yaml"))
    args = parser.parse_args()

    # 读取配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    base_url = config["ollama_base_url"]
    num_ctx = config.get("ollama_options", {}).get("num_ctx", 8192)

    # 检查 Ollama
    host = base_url.replace('/v1', '')
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        assert r.status_code == 200
    except:
        print(f"[ERR] Ollama 不可达: {base_url}")
        sys.exit(1)
    print(f"[OK] Ollama 在线")

    # 加载预测
    if not PREDICTIONS_FILE.exists():
        print(f"[ERR] 预测文件不存在: {PREDICTIONS_FILE}")
        sys.exit(1)
    predictions = []
    with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    print(f"[OK] 加载 {len(predictions)} 条预测")

    # 逐 Judge 跑校准
    output_files = []
    for judge_model in args.judge:
        out = run_calibration(judge_model, predictions, base_url, num_ctx)
        output_files.append(out)

    # 对比
    print_comparison(output_files)


if __name__ == "__main__":
    main()
