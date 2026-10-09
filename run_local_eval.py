#!/usr/bin/env python3
"""
run_local_eval.py - ChineseSimpleQA 全本地评测入口
两阶段流水线：Predict → Judge，支持断点续跑。

用法:
  # Smoke test (5题, 单模型)
  python run_local_eval.py --limit 5 --models qwen2.5:7b

  # 全量 Predict 阶段
  python run_local_eval.py --phase predict

  # 全量 Judge 阶段
  python run_local_eval.py --phase judge

  # 完整流水线
  python run_local_eval.py

  # 切换对照 Judge
  python run_local_eval.py --phase judge --judge llama3.2:3b

  # 只跑聚合
  python run_local_eval.py --phase aggregate
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml
from openai import OpenAI
from tqdm import tqdm

# ============================================================
# 常量
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config_local.yaml"

SYSTEM_MESSAGE = "你是一个智能助手。"
JUDGE_SYSTEM_MESSAGE = "你是一个智能助手，请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。"

# 评分模板 - 与官方 chinese_simpleqa_eval.py 中的 GRADER_TEMPLATE 完全一致
# 不修改原仓库文件，此处独立引用
GRADER_TEMPLATE = """
请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。您的任务是将结果评定为：【正确】、【错误】或【未尝试】。

首先，我们将列出每个评定类别的示例，然后请您对新问题的预测答案进行评定。
以下是【正确】的答复示例：
```
问题：贝拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：Malia Obama and Sasha Obama
模型预测2：玛丽亚和萨莎
模型预测3：大多数人会说是玛丽亚和萨莎，但我不确定，需要再确认
模型预测4：巴拉克·奥巴马有两个女儿，她们分别是玛丽亚·安和娜塔莎·玛丽安，但通常称作玛丽亚·奥巴马和萨莎·奥巴马。玛丽亚出生于1998年7月4日，萨莎出生于2001年6月10日。
```
这些答复均为【正确】，因为：
    - 完整地包含了标准答案中的重要信息。
    - 不包含任何与标准答案矛盾的信息。
    - 只关注语义内容，中英文，大小写、标点、语法和顺序不重要。
    - 答复中出现模糊语句或猜测是可以接受的，前提是包含了标准答案且不含有不正确信息或矛盾。

以下是【错误】的答复示例：
```
问题：贝拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：玛丽亚
模型预测2：玛丽亚、萨莎和苏珊
模型预测3：巴拉克·奥巴马没有孩子
模型预测4：我认为是玛丽亚和萨莎。或者是玛丽亚和杰基。或者是乔伊和玛丽亚。
模型预测5：虽然我不知道他们的确切名字，但能说出巴拉克·奥巴马有三个孩子。
模型预测6：你可能是想说贝茜和奥利维亚。不过您应通过最新的参考资料确认详细信息。那是正确的答案吗？
```
这些答复均为【错误】，因为：
    - 答复中包含与标准答案矛盾的事实陈述。即使在陈述中略带保留（例如："可能是"，"虽然我不确定，但我认为"），也视为错误。

以下是【未尝试】的答复示例：
```
问题：贝拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：我不知道。
模型预测2：我需要更多关于您所指奥巴马的上下文。
模型预测3：不查阅网络我无法回答这个问题，不过我知道巴拉克·奥巴马有两个孩子。
模型预测4：巴拉克·奥巴马有两个孩子。我知道其中一个叫玛丽亚，但我不确定另一个的名字。
```
这些答复均为【未尝试】，因为：
    - 没有包含标准答案中的重要信息。
    - 回复中没有与标准答案矛盾的陈述。

另外注意以下几点：
- 对于标准答案为数字的问题，预测答案应和标准答案一致。例如，考虑问题"金山铁路黄浦江特大桥的全长是多少米？"，标准答案为"3518.17"：
    - 预测答案"3518"、"3518.1"、"3518.17"均为【正确】。
    - 预测答案"3520"和"3600"均为【错误】。 
    - 预测答案"大约3500米"和"超过3000米"被视为【未尝试】，因为它们既不确认也不与标准答案矛盾。
- 如果标准答案包含比问题更多的信息，预测答案只需包含问题中提到的信息。
    - 例如，考虑问题"菱镁矿的主要化学成分是什么？"标准答案为"碳酸镁（MgCO3）"。"碳酸镁"或"MgCO3"均视为【正确】答案。
- 如果从问题中明显可以推断出预测答案省略的信息，那么算作正确。
    - 例如，问题"巴鲁米尼的努拉吉遗迹在1997年被联合国教科文组织列为世界文化遗产，那么这遗址在哪个地区？"标准答案为"意大利撒丁岛"，预测答案"撒丁岛"被视为【正确】。
- 如果能明显看出名字翻译版本不同但是是同一个人也认为正确。
    - 例如，如果标准答案是"Robinson"，那么回答鲁滨逊或者鲁滨孙均正确。

下面是一个新的问题示例。请只回复A、B、C之一，不要道歉或纠正自己的错误，只需要评估该回答。
```
问题: {question}
正确答案: {target}
预测答案: {predicted_answer}
```

将此新问题的预测答案评定为以下之一：
A:【正确】
B:【错误】
C:【未尝试】

只返回字母"A"、"B"或"C"，无须添加其他文本。
""".strip()


# ============================================================
# 校准版评分模板（prompt_version: calibrated）
# 相对官方模板的改进（依据 outputs/local_eval/calibration 的校准实验）：
#   1. 开头用英文显式声明 A/B/C 定义
#   2. 新增 9 条 CRITICAL BOUNDARY RULES 写死边界
#   3. 新增 "KEY DISTINCTION: B vs C" 段落，明确"答错 ≠ 未尝试"
#   4. 规则 6：否认前提但提到关键词 → 不判 A
#   5. 规则 7：异体字/音译变体 → A
#   6. System prompt 改英文
#   7. 结尾强调只输出单个字母
# 实验结论：该模板显著修复了 bonsai 的"C 滥用/把拒答判成 B"问题；
#           对 deepseek-r1:14b 无净提升（prompt 不是其瓶颈），故默认仍用官方模板。
# ============================================================

CALIBRATED_GRADER_TEMPLATE = """
You are a strict grader. Your task is to evaluate a candidate answer against a reference answer and assign exactly one grade: A, B, or C.

=== GRADING RULES (READ CAREFULLY) ===

A = CORRECT: The candidate answer directly answers the question and is factually correct relative to the reference answer.

B = INCORRECT: The candidate attempts to answer the question, but the answer is factually wrong, contradicts the reference, or provides a different entity/value than the reference.

C = NOT ATTEMPTED: The candidate does NOT actually answer the question. This includes: explicit refusal, saying "I don't know", saying "I cannot find this information", providing no substantive answer, or giving an incomplete/truncated response that contains no answer.

=== CRITICAL BOUNDARY RULES ===

1. Correct answer + extra correct information → A
2. Correct answer + irrelevant but non-contradictory extra information → A
3. Correct answer + a factual error that changes or contradicts the core answer → B
4. Wrong answer (model attempted but got it wrong) → B, NOT C
5. Refusal / "I don't know" / "I cannot find" / no substantive answer → C
6. Mentioning the correct keyword while DENYING the premise does NOT make it A → B
7. Character variants of the same name (e.g. 喻浩/喻皓, different transliterations of the same person) → A
8. Numerical precision: if the reference is "3518.17", answers "3518", "3518.1", "3518.17" are all A; "3520" or "3600" are B; "approximately 3500" is C.
9. If the reference contains more information than the question asks, the candidate only needs to answer what the question asks.

=== EXAMPLES ===

以下是【正确】(A) 的答复示例：
```
问题：贝拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：Malia Obama and Sasha Obama
模型预测2：玛丽亚和萨莎
模型预测3：大多数人会说是玛丽亚和萨莎，但我不确定，需要再确认
模型预测4：巴拉克·奥巴马有两个女儿，她们分别是玛丽亚·安和娜塔莎·玛丽安，但通常称作玛丽亚·奥巴马和萨莎·奥巴马。
```
这些答复均为 A【正确】，因为：
    - 完整地包含了标准答案中的重要信息。
    - 不包含任何与标准答案矛盾的信息。
    - 只关注语义内容，中英文、大小写、标点、语法和顺序不重要。
    - 答复中出现模糊语句或猜测是可以接受的，前提是包含了标准答案且不含有不正确信息或矛盾。

以下是【错误】(B) 的答复示例：
```
问题：贝拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：玛丽亚
模型预测2：玛丽亚、萨莎和苏珊
模型预测3：巴拉克·奥巴马没有孩子
模型预测4：我认为是玛丽亚和萨莎。或者是玛丽亚和杰基。或者是乔伊和玛丽亚。
模型预测5：虽然我不知道他们的确切名字，但能说出巴拉克·奥巴马有三个孩子。
模型预测6：你可能是想说贝茜和奥利维亚。不过您应通过最新的参考资料确认详细信息。
```
这些答复均为 B【错误】，因为：
    - 答复中包含与标准答案矛盾的事实陈述。即使在陈述中略带保留（例如："可能是"，"虽然我不确定，但我认为"），也视为错误。

以下是【未尝试】(C) 的答复示例：
```
问题：贝拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：我不知道。
模型预测2：我需要更多关于您所指奥巴马的上下文。
模型预测3：不查阅网络我无法回答这个问题，不过我知道巴拉克·奥巴马有两个孩子。
模型预测4：巴拉克·奥巴马有两个孩子。我知道其中一个叫玛丽亚，但我不确定另一个的名字。
```
这些答复均为 C【未尝试】，因为：
    - 没有包含标准答案中的重要信息。
    - 回复中没有与标准答案矛盾的陈述。
    - 模型明确表示无法回答或信息不足。

=== KEY DISTINCTION: B vs C ===
- If the model GIVES an answer but the answer is WRONG → B (not C!)
- If the model says "I don't know" / "I cannot find" / refuses to answer → C
- If the model's output is truncated and contains no answer → C
- A wrong answer is still an ATTEMPT. Only non-answers are C.

=== YOUR TASK ===

问题: {question}
正确答案: {target}
预测答案: {predicted_answer}

Grade this prediction as exactly one of:
A:【正确】
B:【错误】
C:【未尝试】

Respond with ONLY the letter A, B, or C. No explanation needed.
""".strip()

# 校准版配套 system prompt（英文，对 bonsai 之类模型更友好）
CALIBRATED_JUDGE_SYSTEM_MESSAGE = (
    "You are a precise grading assistant. "
    "Follow the grading rules exactly and output only A, B, or C."
)


def resolve_grader_template(prompt_version: str):
    """按 prompt_version 返回 (grader_template, system_message)"""
    if prompt_version == "calibrated":
        return CALIBRATED_GRADER_TEMPLATE, CALIBRATED_JUDGE_SYSTEM_MESSAGE
    return GRADER_TEMPLATE, JUDGE_SYSTEM_MESSAGE


# ============================================================
# 工具函数
# ============================================================

def sanitize_model_name(model: str) -> str:
    """将模型名转为安全文件名: deepseek-v2:16b -> deepseek-v2_16b"""
    return re.sub(r'[^a-zA-Z0-9._-]', '_', model)


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_dataset(data_path: Path, limit: int = None, seed: int = 42) -> list[dict]:
    """加载 JSONL 数据集，返回 list of dict"""
    items = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if limit and limit < len(items):
        import random
        rng = random.Random(seed)
        items = rng.sample(items, limit)
    return items


def get_done_ids(jsonl_path: Path) -> set:
    """读取已有 JSONL，返回已完成的 id 集合（用于断点续跑）"""
    done = set()
    if jsonl_path.exists():
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if obj.get("id"):
                            done.add(obj["id"])
                    except json.JSONDecodeError:
                        continue
    return done


def append_jsonl(jsonl_path: Path, record: dict):
    """追加一行 JSON 并立即 flush"""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()


def parse_grade(judge_raw: str) -> str:
    """
    从 Judge 原始输出中解析 A/B/C 判定。
    适配思考模型（deepseek-r1 等）：先剥离 think 块，再取最后一个独立 A/B/C。
    """
    THINK_OPEN = "<" + "think>"
    THINK_CLOSE = "</" + "think>"
    # Step 1: 剥离思考块（闭合）
    pattern_closed = re.escape(THINK_OPEN) + r".*?" + re.escape(THINK_CLOSE)
    text = re.sub(pattern_closed, "", judge_raw, flags=re.DOTALL)
    # 也处理未闭合的 think 块（模型被截断时）
    pattern_open = re.escape(THINK_OPEN) + r".*"
    text = re.sub(pattern_open, "", text, flags=re.DOTALL)
    text = text.strip()

    # Step 2: 在剥离后的文本中找所有独立的 A/B/C，取最后一个
    matches = re.findall(r"\b([ABC])\b", text)
    if matches:
        return matches[-1]

    # Step 3: 回退 - 全文（含思考块）取最后一个
    matches_all = re.findall(r"\b([ABC])\b", judge_raw)
    if matches_all:
        return matches_all[-1]

    # 兜底：无匹配返回 C（未尝试）
    return "C"


def ollama_unload(base_url: str, model: str):
    """通过 Ollama 原生 API 强制卸载模型释放 VRAM"""
    # base_url 形如 http://localhost:11434/v1, 取 host 部分
    host = base_url.replace('/v1', '').replace('/v1/', '')
    try:
        requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0},
            timeout=10
        )
        print(f"  [VRAM] 已卸载模型: {model}")
    except Exception as e:
        print(f"  [VRAM] 卸载 {model} 失败(非致命): {e}")


def ollama_check_alive(base_url: str) -> bool:
    """检查 Ollama 服务是否在线"""
    host = base_url.replace('/v1', '').replace('/v1/', '')
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def call_ollama_chat(client: OpenAI, model: str, messages: list[dict],
                     temperature: float = 0.0, max_tokens: int = 2048,
                     timeout: int = 180, max_retries: int = 3,
                     retry_backoff: int = 5, num_ctx: int = 8192) -> str | None:
    """调用 Ollama OpenAI-compatible API，带重试。返回 content 或 None。"""
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
            # content 为 None，重试
            print(f"    [WARN] 模型返回 None (attempt {attempt+1})")
        except Exception as e:
            print(f"    [ERR] {type(e).__name__}: {e} (attempt {attempt+1}/{max_retries})")
        if attempt < max_retries - 1:
            time.sleep(retry_backoff * (attempt + 1))
    return None


# ============================================================
# Phase 1: Predict
# ============================================================

def phase_predict(config: dict, dataset: list[dict], output_dir: Path,
                  models_filter: list[str] = None):
    """逐模型串行预测，每个模型跑完卸载再跑下一个"""
    base_url = config["ollama_base_url"]
    all_models = config["target_models"]
    if models_filter:
        all_models = [m for m in all_models if m in models_filter]
        if not all_models:
            print(f"[ERR] --models 过滤后无匹配模型。可用: {config['target_models']}")
            return

    gen_cfg = config.get("generation", {})
    default_temp = gen_cfg.get("temperature", 0.0)
    default_max_tokens = gen_cfg.get("max_tokens", 2048)
    overrides = gen_cfg.get("per_model_overrides", {})
    timeout = config.get("timeout", {}).get("predict", 180)
    max_retries = config.get("max_retries", 3)
    retry_backoff = config.get("retry_backoff", 5)
    num_ctx = config.get("ollama_options", {}).get("num_ctx", 8192)

    predict_dir = output_dir / "predictions"
    predict_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key="ollama", base_url=base_url)

    for model in all_models:
        model_safe = sanitize_model_name(model)
        pred_file = predict_dir / f"{model_safe}.jsonl"
        done_ids = get_done_ids(pred_file)
        todo = [item for item in dataset if item["id"] not in done_ids]

        print(f"\n{'='*60}")
        print(f"[PREDICT] 模型: {model}")
        print(f"  已完成: {len(done_ids)}, 待跑: {len(todo)}, 总计: {len(dataset)}")
        print(f"  输出: {pred_file}")
        print(f"{'='*60}")

        if not todo:
            print("  全部已完成，跳过。")
            ollama_unload(base_url, model)
            continue

        # 模型级参数覆盖
        model_override = overrides.get(model, {})
        temp = model_override.get("temperature", default_temp)
        max_tok = model_override.get("max_tokens", default_max_tokens)

        t0 = time.time()
        success_count = 0
        fail_count = 0

        for item in tqdm(todo, desc=f"  {model}", unit="q"):
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": item["question"]},
            ]
            response = call_ollama_chat(
                client, model, messages,
                temperature=temp, max_tokens=max_tok,
                timeout=timeout, max_retries=max_retries,
                retry_backoff=retry_backoff
            )

            record = {
                "id": item["id"],
                "model": model,
                "primary_category": item.get("primary_category", ""),
                "secondary_category": item.get("secondary_category", ""),
                "question": item["question"],
                "answer": item["answer"],
                "model_output": response if response else "",
                "status": "ok" if response else "failed",
                "timestamp": datetime.now().isoformat(),
            }
            append_jsonl(pred_file, record)

            if response:
                success_count += 1
            else:
                fail_count += 1

        elapsed = time.time() - t0
        print(f"  完成: 成功={success_count}, 失败={fail_count}, "
              f"耗时={elapsed/60:.1f}min, 平均={elapsed/max(success_count+fail_count,1):.1f}s/题")

        # 卸载模型释放 VRAM
        ollama_unload(base_url, model)


# ============================================================
# Phase 2: Judge
# ============================================================

def phase_judge(config: dict, dataset: list[dict], output_dir: Path,
                judge_model: str = None, models_filter: list[str] = None,
                prompt_version: str = None):
    """加载 Judge 模型，逐预测文件评分"""
    base_url = config["ollama_base_url"]
    judge_cfg = config.get("judge", {})

    if judge_model is None:
        judge_model = judge_cfg.get("primary", "qwen2.5:7b")

    if prompt_version is None:
        prompt_version = judge_cfg.get("prompt_version", "official")

    grader_tmpl, judge_sys_msg = resolve_grader_template(prompt_version)

    judge_temp = judge_cfg.get("temperature", 0.0)
    judge_max_tokens = judge_cfg.get("max_tokens", 16)
    timeout = config.get("timeout", {}).get("judge", 60)
    max_retries = config.get("max_retries", 3)
    retry_backoff = config.get("retry_backoff", 5)
    num_ctx = config.get("ollama_options", {}).get("num_ctx", 8192)

    predict_dir = output_dir / "predictions"
    review_dir = output_dir / "reviews"
    review_dir.mkdir(parents=True, exist_ok=True)

    if not predict_dir.exists():
        print("[ERR] predictions 目录不存在，请先跑 --phase predict")
        return

    # 确定要 judge 的模型文件
    pred_files = sorted(predict_dir.glob("*.jsonl"))
    if models_filter:
        filter_safe = {sanitize_model_name(m) for m in models_filter}
        pred_files = [f for f in pred_files if f.stem in filter_safe]

    if not pred_files:
        print("[ERR] 无匹配的预测文件")
        return

    print(f"\n{'='*60}")
    print(f"[JUDGE] Judge 模型: {judge_model}")
    print(f"  Prompt 版本: {prompt_version}")
    print(f"  num_ctx: {num_ctx}")
    print(f"  待评文件: {[f.name for f in pred_files]}")
    print(f"{'='*60}")

    client = OpenAI(api_key="ollama", base_url=base_url)
    judge_safe = sanitize_model_name(judge_model)

    for pred_file in pred_files:
        model_safe = pred_file.stem
        # 文件名带 prompt 版本，避免不同 Prompt 的结果互相覆盖
        review_file = review_dir / f"{model_safe}__judged_by__{judge_safe}__{prompt_version}.jsonl"
        done_ids = get_done_ids(review_file)

        # 读取预测
        predictions = []
        with open(pred_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        predictions.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        todo = [p for p in predictions if p["id"] not in done_ids and p.get("status") == "ok"]
        skipped_failed = len([p for p in predictions if p.get("status") != "ok"])

        print(f"\n  [{model_safe}] 已评: {len(done_ids)}, 待评: {len(todo)}, "
              f"预测失败跳过: {skipped_failed}")

        if not todo:
            print("    全部已完成，跳过。")
            continue

        t0 = time.time()
        for pred in tqdm(todo, desc=f"    judge:{model_safe}", unit="q"):
            grader_prompt = grader_tmpl.format(
                question=pred["question"],
                target=pred["answer"],
                predicted_answer=pred["model_output"],
            )
            messages = [
                {"role": "system", "content": judge_sys_msg},
                {"role": "user", "content": grader_prompt},
            ]
            judge_raw = call_ollama_chat(
                client, judge_model, messages,
                temperature=judge_temp, max_tokens=judge_max_tokens,
                timeout=timeout, max_retries=max_retries,
                retry_backoff=retry_backoff, num_ctx=num_ctx
            )

            # 解析 A/B/C（适配思考模型：先剥离 <think> 块，再取最后一个匹配）
            grade_letter = "C"  # 默认未尝试
            if judge_raw:
                grade_letter = parse_grade(judge_raw)

            record = {
                "id": pred["id"],
                "model": pred["model"],
                "judge_model": judge_model,
                "prompt_version": prompt_version,
                "primary_category": pred.get("primary_category", ""),
                "question": pred["question"],
                "answer": pred["answer"],
                "model_output": pred["model_output"],
                "judge_raw": judge_raw if judge_raw else "",
                "grade": grade_letter,
                "is_correct": grade_letter == "A",
                "is_incorrect": grade_letter == "B",
                "is_not_attempted": grade_letter == "C",
                "timestamp": datetime.now().isoformat(),
            }
            append_jsonl(review_file, record)

        elapsed = time.time() - t0
        print(f"    完成, 耗时={elapsed/60:.1f}min")

    # Judge 结束，卸载
    ollama_unload(base_url, judge_model)


# ============================================================
# Phase 3: Aggregate
# ============================================================

def phase_aggregate(config: dict, output_dir: Path, judge_model: str = None,
                    prompt_version: str = None):
    """读取 review JSONL，计算指标，输出 leaderboard"""
    judge_cfg = config.get("judge", {})
    if judge_model is None:
        judge_model = judge_cfg.get("primary", "qwen2.5:7b")
    if prompt_version is None:
        prompt_version = judge_cfg.get("prompt_version", "official")
    judge_safe = sanitize_model_name(judge_model)

    review_dir = output_dir / "reviews"
    if not review_dir.exists():
        print("[ERR] reviews 目录不存在，请先跑 --phase judge")
        return

    # 兼容两种文件名：
    #   新: {model}__judged_by__{judge}__{prompt_version}.jsonl
    #   旧: {model}__judged_by__{judge}.jsonl
    review_files = sorted(review_dir.glob(f"*__judged_by__{judge_safe}__{prompt_version}.jsonl"))
    if not review_files:
        legacy = sorted(review_dir.glob(f"*__judged_by__{judge_safe}.jsonl"))
        # 排除带 prompt 版本后缀的
        legacy = [f for f in legacy
                  if "__official" not in f.stem and "__calibrated" not in f.stem]
        if legacy:
            print(f"  [WARN] 未找到 prompt_version={prompt_version} 的结果，"
                  f"回退使用旧命名文件（无 prompt 版本标记）")
            review_files = legacy

    if not review_files:
        print(f"[ERR] 无匹配 judge={judge_model} prompt_version={prompt_version} 的 review 文件")
        return

    print(f"\n{'='*60}")
    print(f"[AGGREGATE] Judge: {judge_model}  (prompt_version={prompt_version})")
    print(f"{'='*60}")

    leaderboard_rows = []

    for rf in review_files:
        model_safe = rf.stem.split("__judged_by__")[0]
        records = []
        with open(rf, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not records:
            continue

        # 总体指标
        total = len(records)
        correct = sum(1 for r in records if r.get("is_correct"))
        incorrect = sum(1 for r in records if r.get("is_incorrect"))
        not_attempted = sum(1 for r in records if r.get("is_not_attempted"))
        given_attempted = correct + incorrect
        acc = correct / total if total > 0 else 0
        acc_given_attempted = correct / given_attempted if given_attempted > 0 else 0
        f1 = (2 * acc_given_attempted * acc / (acc_given_attempted + acc)
              if (acc_given_attempted + acc) > 0 else 0)

        print(f"\n  [{model_safe}] n={total}")
        print(f"    Correct: {correct} ({acc*100:.1f}%)")
        print(f"    Incorrect: {incorrect} ({incorrect/total*100:.1f}%)")
        print(f"    Not Attempted: {not_attempted} ({not_attempted/total*100:.1f}%)")
        print(f"    Acc(given attempted): {acc_given_attempted*100:.1f}%")
        print(f"    F1: {f1*100:.1f}%")

        # 按 primary_category 分组
        categories = {}
        for r in records:
            cat = r.get("primary_category", "未知")
            if cat not in categories:
                categories[cat] = {"total": 0, "correct": 0, "incorrect": 0, "not_attempted": 0}
            categories[cat]["total"] += 1
            if r.get("is_correct"):
                categories[cat]["correct"] += 1
            elif r.get("is_incorrect"):
                categories[cat]["incorrect"] += 1
            else:
                categories[cat]["not_attempted"] += 1

        print(f"    {'类别':<20} {'n':>4} {'Correct%':>9} {'Incorr%':>8} {'N/A%':>6} {'F1%':>6}")
        print(f"    {'-'*58}")
        for cat, v in sorted(categories.items(), key=lambda x: -x[1]["total"]):
            c_acc = v["correct"] / v["total"] * 100 if v["total"] > 0 else 0
            c_inc = v["incorrect"] / v["total"] * 100 if v["total"] > 0 else 0
            c_na = v["not_attempted"] / v["total"] * 100 if v["total"] > 0 else 0
            c_ga = v["correct"] / (v["correct"] + v["incorrect"]) * 100 if (v["correct"] + v["incorrect"]) > 0 else 0
            c_f1 = (2 * c_ga * c_acc / (c_ga + c_acc)) if (c_ga + c_acc) > 0 else 0
            print(f"    {cat:<20} {v['total']:>4} {c_acc:>8.1f}% {c_inc:>7.1f}% {c_na:>5.1f}% {c_f1:>5.1f}%")

        leaderboard_rows.append({
            "model": model_safe,
            "judge": judge_model,
            "n": total,
            "correct": acc,
            "incorrect": incorrect / total if total > 0 else 0,
            "not_attempted": not_attempted / total if total > 0 else 0,
            "acc_given_attempted": acc_given_attempted,
            "f1": f1,
        })

    # 写 leaderboard CSV
    if leaderboard_rows:
        import csv
        lb_path = output_dir / f"leaderboard__judge_{judge_safe}__{prompt_version}.csv"
        with open(lb_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=leaderboard_rows[0].keys())
            writer.writeheader()
            writer.writerows(leaderboard_rows)
        print(f"\n  Leaderboard 已写入: {lb_path}")

        # 打印汇总表
        print(f"\n{'='*60}")
        print(f"  {'模型':<22} {'Correct%':>9} {'AccAtt%':>8} {'F1%':>6}")
        print(f"  {'-'*50}")
        for row in sorted(leaderboard_rows, key=lambda x: -x["f1"]):
            print(f"  {row['model']:<22} {row['correct']*100:>8.1f}% "
                  f"{row['acc_given_attempted']*100:>7.1f}% {row['f1']*100:>5.1f}%")
        print(f"{'='*60}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ChineseSimpleQA 本地评测")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="配置文件路径 (默认: config_local.yaml)")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["predict", "judge", "aggregate", "all"],
                        help="运行阶段 (默认: all = predict→judge→aggregate)")
    parser.add_argument("--limit", type=int, default=None,
                        help="覆盖配置中的 limit (抽样题数)")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="只跑指定 target 模型 (空格分隔)")
    parser.add_argument("--judge", type=str, default=None,
                        help="覆盖 judge 模型 (默认用 config 中的 primary)")
    parser.add_argument("--prompt-version", type=str, default=None,
                        choices=["official", "calibrated"],
                        help="Judge 评分模板版本 (默认用 config 中 judge.prompt_version，缺省 official)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="覆盖输出目录")
    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERR] 配置文件不存在: {config_path}")
        sys.exit(1)
    config = load_config(config_path)

    # 覆盖参数
    if args.limit is not None:
        config.setdefault("dataset", {})["limit"] = args.limit
    if args.output_dir:
        config["output_dir"] = args.output_dir

    # 解析路径
    dataset_path = PROJECT_ROOT / config.get("dataset", {}).get("path", "data/chinese_simpleqa.jsonl")
    output_dir = Path(config.get("output_dir", "./outputs/local_eval"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    limit = config.get("dataset", {}).get("limit", None)

    # 检查 Ollama
    base_url = config["ollama_base_url"]
    if not ollama_check_alive(base_url):
        print(f"[ERR] Ollama 服务不可达: {base_url}")
        print("  请确认 Ollama 已启动 (ollama serve)")
        sys.exit(1)
    print(f"[OK] Ollama 在线: {base_url}")

    # 加载数据集
    if not dataset_path.exists():
        print(f"[ERR] 数据集不存在: {dataset_path}")
        sys.exit(1)
    dataset = load_dataset(dataset_path, limit=limit)
    print(f"[OK] 数据集: {len(dataset)} 题 (limit={limit})")

    # 保存本次运行配置快照
    run_config_path = output_dir / "run_config.yaml"
    with open(run_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    judge_model = args.judge
    prompt_version = args.prompt_version

    # 执行阶段
    if args.phase in ("predict", "all"):
        phase_predict(config, dataset, output_dir, models_filter=args.models)

    if args.phase in ("judge", "all"):
        phase_judge(config, dataset, output_dir, judge_model=judge_model,
                    models_filter=args.models, prompt_version=prompt_version)

    if args.phase in ("aggregate", "all"):
        phase_aggregate(config, output_dir, judge_model=judge_model,
                        prompt_version=prompt_version)

    print("\n[DONE] 评测完成。")


if __name__ == "__main__":
    main()
