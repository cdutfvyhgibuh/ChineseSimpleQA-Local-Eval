# ChineseSimpleQA-Local-Eval

> **在本地环境中评测中文事实问答模型，并检查 Judge 的判定可靠性**
>
> 本项目在单张 RTX 5080 16GB 上通过 Ollama 评测 3 个不同架构的中文模型。三个模型使用同一款经过人工校准的本地 Judge，对 3000 道题进行评分；预测和判定的原始记录均予以保留，便于复现、复核，以及使用其他 Judge 重新评分。

📊 **[在线查看交互式评测报告 →](https://cdutfvyhgibuh.github.io/ChineseSimpleQA-Local-Eval/)**
（8 张可交互图表：总体指标、分类别热力图、难度分层、模型互补性、Judge 可信度）

---

## English Abstract

This project runs **ChineseSimpleQA**, a 3,000-question Chinese factual QA benchmark, entirely on a single RTX 5080 16GB machine via Ollama to reduce network-related variation.

It evaluates **Qwen2.5 7B Instruct** (Dense/Instruct), **DeepSeek-R1 7B** (Reasoning), and **DeepSeek-V2 16B** (MoE) using the same manually calibrated local judge, **DeepSeek-R1 14B**, for 9,000 judgments.

Before the full run, four candidate judges were compared on 50 questions. Their tendencies toward overly strict or lenient grading and misuse of class C were recorded. Fifteen disputed items were manually reviewed, and five facts were independently checked online. A prompt with explicit boundary rules was A/B tested but did not reduce DeepSeek-R1 14B's total errors, so the official grading template was retained.

**Results (F1):** DeepSeek-V2 16B **39.8%**, Qwen2.5 7B **33.9%**, and DeepSeek-R1 7B **11.3%**. All models missed 50.9% of questions; pairwise agreement fell to 47.5%, while each model also answered some items the others missed.

---

## 1 上游项目本地化：把官方评测搬到本机

本项目使用开源项目 **[LivingFutureLab/ChineseSimpleQA](https://github.com/LivingFutureLab/ChineseSimpleQA)** 提供的数据集与评分口径。为在本机通过 Ollama 完成评测，我检查了上游脚本的输入格式、依赖和评分解析方式，并记录了实际遇到的问题及相应处理。

### 1.1 上游提供的三条评测路径

上游 README 介绍了三种运行方式：

| 路径 | 依赖 | 本地化障碍 |
|---|---|---|
| `simple-evals` 框架（`python -m simple-evals.demo`） | OpenAI 云端 API | 需联网 + API Key，与本项目「排除网络变量」的目标冲突 |
| 独立脚本 `scripts/chinese_simpleqa_easy.py`（运行入口 `judge/chinese_simpleqa_easy.py`） | 需在源码里硬编码 `OPENAI_API_KEY` / `OPENAI_BASE_URL` | 评测逻辑与云端客户端耦合 |
| OpenCompass 框架 | `git clone open-compass` + 按其 config 配置模型 | 框架重、模型接入方式受限，不适合 Ollama 直连 |

这三种方式都把评测程序与特定推理后端或数据格式联系在一起，没有可以直接接收本地模型回答的独立入口。因此，本项目保留数据集和 A/B/C 评分口径，另行实现了面向 Ollama 的执行流程。

### 1.2 上游代码与本地运行环境的差异

以下问题均根据上游代码逐项检查后记录。

**① 数据格式不同：上游读取 CSV，本项目使用 JSONL**

```python
# chinese_simpleqa_eval.py:99
df = pandas.read_csv('chinese_simpleqa.csv')
```

- **硬编码相对路径**：依赖「当前工作目录下正好有个叫 `chinese_simpleqa.csv` 的文件」
- **格式不一致**：本项目使用的是 `data/chinese_simpleqa.jsonl`

**② 字段名不匹配**

上游代码读取的字段名是 `problem`：

```python
# chinese_simpleqa_eval.py:128
sampler._pack_message(content=row.get("problem", ""), role="user")
```

而数据集实际字段是：

```json
{"id": "...", "primary_category": "中华文化", "secondary_category": "中医",
 "question": "...", "answer": "...", "urls": "..."}
```

数据集中没有 `problem` 字段。由于 `row.get("problem", "")` 设置了默认值，程序不会因此报错，而是把空字符串作为问题传给模型，随后继续评分。这样一来，评测可能顺利结束，却无法得到有效结果。

本地版本改用 `question` 字段，并检查每条记录的 id 是否与数据集中的 id 集合一致（见第 10 节），以便尽早发现这类静默错误。

**③ 相对导入与包结构**

```python
# chinese_simpleqa_eval.py:5-6
from . import common
from .types_local import Eval, EvalResult, SamplerBase, SingleEvalResult
```

`from . import ...` 属于包内相对导入，因此该文件需要作为包的一部分导入，不能直接通过 `python chinese_simpleqa_eval.py` 运行。本地版本改用同级模块的直接 `import`，不再依赖原有包结构。

**④ 多余的重量级依赖**

```python
# chinese_simpleqa_eval.py:3-4
import blobfile as bf
import pandas
```

`blobfile` 用于访问对象存储，本地评测不需要该库；当前环境也未安装它，运行时会出现 `ImportError`。对于这份数据，读取和处理也不一定需要 `pandas`。本地流水线使用 `openai`（连接 Ollama 的 OpenAI 兼容端点）、`pyyaml`、`requests` 和 `tqdm`；生成 HTML 报告时另需 `plotly`。

**⑤ 评分正则可能解析出错误类别**

```python
# chinese_simpleqa_eval.py:121
match = re.search(r"(A|B|C)", grading_response)
return match.group(0) if match else "C"
```

这里有两个风险：一是正则返回第一个匹配项；如果 Judge 的回复包含推理文本，取到的字母可能不是最终结论。二是正则也可能匹配英文单词中的字母，例如 `Correct` 中的 `C`。

本项目选用的 Judge 是思考模型 **DeepSeek-R1 14B**，输出可能包含较长的推理文本，因此本地版本调整为：

```python
# 先剥离 <think>…</think> 思考块（含未闭合的截断情形）
# 再取最后一个独立的 \b[ABC]\b 匹配
# 仅当剥离后无匹配时，才回退到全文匹配；最终兜底返回 C
```

在这些本地化调整中，这一项直接关系到评分结果；其他修改主要用于适配数据格式和运行环境。

### 1.3 本地化的处理原则

针对这些问题，可以直接修改上游脚本，也可以保留上游文件并单独实现本地执行流程。本项目采用第二种方式，具体处理如下：

| 原则 | 具体做法 |
|---|---|
| 上游文件保持可对照 | `chinese_simpleqa_eval.py`、`simpleqa_eval.py`、`judge/`、`sampler/` 均不改动，作为口径的权威参照 |
| 评分定义以官方为准 | A/B/C 的定义、数值精度规则、异体字规则、F1 计算公式，全部照搬官方 `GRADER_TEMPLATE` 与官方聚合逻辑 |
| 只在必要处重写 | 数据加载（CSV→JSONL）、字段映射、模型调用（云端→Ollama）、A/B/C 解析、断点续跑、结果落盘 |
| 唯一的上游改动显式声明 | `common.py` 的 `map_with_progress` 默认 `num_threads` 由 `10` 调整为 `1`（本机并发压力）。该文件**不被本地流水线引用**，因此不影响本项目任何评测结果 |

保留上游文件有助于区分官方定义与本项目的实现。复核者可以将两者进行对照，检查本地版本是否沿用了相同的 A/B/C 定义和指标计算方式。

### 1.4 本地运行流程

```text
上游提供：data/chinese_simpleqa.jsonl（3000 题）+ A/B/C 评分口径与 F1 定义
                    │
                    ▼
本项目新增：run_local_eval.py
    Phase 1  Predict   Ollama 逐个加载目标模型 → 3000 题 → predictions/{model}.jsonl
    Phase 2  Judge     加载 Judge 模型 → 逐条判定 → reviews/{model}__judged_by__{judge}.jsonl
    Phase 3  Aggregate 按官方口径聚合 → leaderboard CSV
                    │
                    ▼
本项目新增：make_report.py → report.html（交互式图表）
```

这条流程通过本机 Ollama 的 OpenAI 兼容端点调用模型，评测运行期间不需要访问外网。

## 2 项目缘起

项目最初是为了比较几个可在本机运行的中文模型在事实问答任务上的表现。早期尝试使用在线 API，但网络稳定性、代理和请求限流都会影响请求过程，使得测量结果难以单独反映模型表现。

因此，我把模型推理、Judge、数据集和评测程序都迁移到本地运行。这样可以减少网络因素对实验的干扰，把分析重点放在模型输出和评分流程上。

人工抽查后，我发现本地 Judge 也会产生明显误判。项目的工作重点随之扩大：除了比较目标模型的分数，还需要先检查 Judge 的误判类型，并评估这些错误会怎样影响最终排名。

## 3 硬件与环境约束

| 项 | 配置 |
|---|---|
| GPU | NVIDIA RTX 5080，16 GB VRAM |
| CPU | AMD Ryzen 7 5700X3D |
| RAM | 32 GB DDR4-3200 |
| OS | Windows 10 |
| 推理后端 | Ollama（OpenAI 兼容 API，`http://localhost:11434/v1`） |

16 GB 显存限制了可同时驻留的模型规模。因此，目标模型与 Judge 分阶段运行，不会同时占用显存。

### 3.1 上下文长度设置

项目早期使用过较大的默认上下文窗口（部分模型标签为 131072）。在 16 GB 显存下，这会显著增加运行时占用，并使部分模型层转移到 CPU：

```text
deepseek-r1:14b   35 GB   60% CPU / 40% GPU   131072 context
```

推理速度因此下降。将上下文长度设为 **8192** 后，日志显示模型层可以全部卸载到 GPU：

```text
100% GPU
```

本项目将每道题作为独立请求处理，3000 道题不会累积到同一个上下文中。因此，对于单题输入通常只有一两千字符的 Judge，128K 窗口并无实际必要。将上下文限制在 8K 后，显存占用和推理速度更适合当前硬件。

实测确认（llama-server 日志）：

```text
-c 8192 -np 1 ... offloaded 29/29 layers to GPU
llama_kv_cache: size = 448.00 MiB (8192 cells, 28 layers, 1/1 seqs)
```

各模型的原生上下文上限和 8K 设置下的实测显存占用如下：

| 模型 | 原生上限 `n_ctx_train` | 8K 下实测显存 |
|---|---|---|
| Qwen2.5 7B Instruct | 32,768 | 4,896 MiB |
| DeepSeek-R1 7B | 32,768 | ~4.9 GB |
| DeepSeek-R1 14B（Judge） | 131,072 | **9,922 MiB** |
| Bonsai 27B（对照 Judge） | 262,144 | 4,367 MiB |

## 4 评测架构

### 4.1 三阶段流水线

```text
Phase 1  Predict（逐模型串行）
  加载 Target → 跑完 3000 题 → 保存 predictions → 卸载 → 下一个 Target

Phase 2  Judge（单独加载 Judge）
  加载 Judge → 评该 Target 的全部 predictions → 保存 reviews → 卸载

Phase 3  Aggregate
  读 reviews → 计算指标 → 输出 leaderboard CSV + HTML 报告
```

### 4.2 设计约束

1. **每个阶段中，模型加载后连续处理该阶段的题目。** 不按题目反复加载和卸载权重，以减少额外开销。
2. **并发数设为 1。** 在 16 GB 显存条件下串行调用模型。
3. **Target 和 Judge 分开运行。** 两者不会同时驻留显存。
4. **逐条写入 JSONL 并立即 flush。** 如果任务中断，已完成的记录仍会保存在文件中。

### 4.3 断点续跑

续跑时，程序会读取已有 JSONL 并根据 `id` 跳过已完成的题目，因此任务中断后可以继续运行，而不必重复处理已有记录。

这种机制也有一个需要注意的情况：如果目录中残留早期 smoke test 的判定文件，正式运行时对应题目可能会被跳过，导致结果混合不同批次的数据。为避免这种情况，本项目在全量评测前将 50 题 smoke test 文件改名为 `.smoketest`，与正式结果分开保存。

**输出命名**（不同 Judge / 不同 Prompt 的结果并存不覆盖）：

```text
predictions/{target}.jsonl
reviews/{target}__judged_by__{judge}__{prompt_version}.jsonl
leaderboard__judge_{judge}__{prompt_version}.csv
```

## 5 Judge 提示词与校准

人工复核发现的误判类型，为后续 Judge 提示词校准提供了依据。

### 5.1 为什么检查 Judge Prompt

官方 `chinese_simpleqa_eval.py` 的评分模板定义了 A/B/C 三档：

- **A = 正确**：包含参考答案要点，且不含矛盾信息
- **B = 错误**：包含与参考答案矛盾的事实陈述
- **C = 未尝试**：没有给出答案，也没有矛盾陈述

在 50 题样本上直接使用官方模板时，几个本地候选 Judge 表现出不同的误判倾向：

| 候选 Judge | 偏差方向 | 典型误判 |
|---|---|---|
| Qwen2.5 7B | **偏严** | 答对但补充了信息 → 误判 B；拒答 → 误判 B 而非 C |
| DeepSeek-R1 7B | **偏松** | 答错 → 误判 A |
| DeepSeek-R1 14B | 偏严（较轻） | 对"已答对但补充无关信息"的题判 B |
| Bonsai 27B | **偏松 + C 滥用** | 答错 → 误判 C；提到关键词即判 A |

偏严和偏松对最终分数的影响不同：前者可能低估目标模型，后者则可能抬高分数。若 Judge 将错误答案判成“未尝试”，目标模型的错误就不会按预期计入评分，尤其可能影响拒答较多的模型。

### 5.2 校准版 Prompt 的改动

根据这些误判案例，我编写了一份将类别边界写得更明确的校准版模板，主要改动如下：

1. **开头用英文显式声明 A/B/C 定义**（对 Bonsai 这类模型英文指令更稳定）
2. **9 条 CRITICAL BOUNDARY RULES** 把边界写死，例如：
   - 规则 3：正确答案 + 与核心答案矛盾的事实错误 → B
   - 规则 4：**答错（模型尝试了但答错）→ B，不是 C**
   - 规则 6：**否认题目前提、但仍提到关键词 → 不算 A**
   - 规则 7：**人名异体字 / 不同音译 → A**（如 喻浩 / 喻皓）
   - 规则 8：数值精度规则（参考值 3518.17，答 3518 / 3518.1 / 3518.17 均为 A；答 3520 为 B；答"大约 3500"为 C）
3. **单独一段 `KEY DISTINCTION: B vs C`**，明确"答错 ≠ 未尝试"
4. **结尾强调只输出单个字母**，降低解析成本

规则 3、4、6 和 7 均来自人工核验中出现过的具体误判。校准版试图把这些边界情况转成明确规则，再通过同一批样本检查修改是否有效。

### 5.3 模板集中定义

校准版模板只在 `run_local_eval.py` 中定义，校准脚本通过 import 复用，以免两处副本出现差异：

```python
from run_local_eval import CALIBRATED_GRADER_TEMPLATE, CALIBRATED_JUDGE_SYSTEM_MESSAGE
```

修改 Prompt 时只需更新这一处定义，两个脚本便会使用同一模板。

## 6 人工复核

自动判定仍需人工抽查，尤其是不同 Judge 给出不同类别的题目。

### 6.1 方法

从 50 题 smoke test 中，选取四个候选 Judge 判定存在分歧的 15 道题，逐题人工核验。
判定依据严格对齐官方口径：A = 含参考答案要点且无矛盾；B = 含矛盾陈述或给出不同实体/数值；C = 未答且无矛盾。

### 6.2 争议题的事实核验

人工判定中有 5 道题的答案事实需要外部确认，不能仅凭印象裁决。下表列出这些题目的最终真值与判定依据：

| 题号 | 核实后真值 | 依据 |
|---|---|---|
| #16 | B | 卢梭《论人类不平等的起源和基础》确为 **1755 年** 4 月初版于阿姆斯特丹；模型答"1754年"与参考答案矛盾 |
| #20 | B | 1954 年被伊朗当局审判的是**伊朗人民党（图德党）**；模型给出"伊朗民族解放阵线"这一不同组织 |
| #42 | B | 书目数据库存储的是**二次文献**；模型答"元数据"，给出了不同概念 |
| #44 | B | 金斗瓮在广东、香港俗称**金塔**；模型答"猪笼草" |
| #50 | B | 释行真确为少林寺**第三十二代**嫡传弟子；模型称其"并非少林寺嫡传弟子" |

核实来源：[卢梭著作（百度百科）](https://baike.baidu.com/item/%E8%AE%BA%E4%BA%BA%E7%B1%BB%E4%B8%8D%E5%B9%B3%E7%AD%89%E7%9A%84%E8%B5%B7%E6%BA%90%E5%92%8C%E5%9F%BA%E7%A1%80/5880820) ·
[金斗瓮（维基百科）](https://zh.wikipedia.org/zh-hans/%E9%87%91%E6%96%97%E7%94%95) ·
[释行真（百度百科）](https://baike.baidu.com/item/%E9%87%8A%E8%A1%8C%E7%9C%9F/3793721) ·
[伊朗人民党（维基百科）](https://zh.wikipedia.org/zh-hans/%E4%BC%8A%E6%9C%97%E4%BA%BA%E6%B0%91%E5%85%9A)

这 5 道题的真值均以外部资料为准，而非依据印象裁决。这也说明人工标注同样可能出错，对争议事实做外部核查是必要的。最终采用的 15 题真值表保存在 `analyze_calibration.py` 的 `HUMAN_GROUND_TRUTH` 中，便于检查和复现。

## 7 四个 Judge 的横向对比

四个候选 Judge 使用同一批 50 题和同一个目标模型（Qwen2.5 7B）进行评分，结果如下：

| Judge | A | B | C | 报告的 Correct% | 人工核验误判 |
|---|---|---|---|---|---|
| Qwen2.5 7B | 13 | 37 | 0 | 26.0% | 偏严，A 明显偏低 |
| DeepSeek-R1 7B | 20 | 29 | 1 | 40.0% | 偏松，答错判成 A |
| **DeepSeek-R1 14B** | 16 | 31 | 3 | 32.0% | **2 / 15（13.3%）** |
| Bonsai 27B | 17 | 28 | 5 | 34.0% | **4 / 15（26.7%）** |

Qwen2.5 7B 在这 50 题中一次也没有使用 C 类，表明它未能正确识别样本中的未尝试回答。DeepSeek-R1 7B 的 Correct% 最高，但人工复核发现，它把部分错误答案判成 A，因此这个比例不能单独作为 Judge 可靠性的依据。

这些结果表明，比较候选 Judge 时，除了它报告的分数，还需要检查各类误判的频率和方向。

### 7.1 推理速度

| Judge | 单题耗时 | 9000 条预计 |
|---|---|---|
| Qwen2.5 7B | ~0.6 s | ~1.5 h |
| DeepSeek-R1 7B | ~1.7 s | ~4.3 h |
| **DeepSeek-R1 14B** | **~3.2 s** | **~8 h** |
| Bonsai 27B | ~10 s | ~25 h |

全量评测最终采用 **DeepSeek-R1 14B**，主要考虑以下因素：

1. **人工复核结果**：在 15 道核验题中，原版 DeepSeek-R1 14B 误判 2 题，Bonsai 27B 误判 4 题。
2. **运行时间**：完成 9000 条判定预计需要约 8 小时，Bonsai 约需 25 小时。
3. **误判倾向**：R1-14B 略偏严，可能低估目标模型的分数；Bonsai 的偏松问题则可能抬高分数。对于本次比较，后者更容易直接影响模型排名的解释。

## 8 校准实验

### 8.1 实验问题

实验要区分剩余误判主要来自 Prompt 边界不够明确，还是来自 Judge 本身的判别能力。如果问题主要在提示词，继续调整模板可能有帮助；如果来自模型本身，则需要考虑更换 Judge，或在结果中记录其偏差。

### 8.2 实验方法

使用第 5 节介绍的校准版 Prompt，重新评测 **DeepSeek-R1 14B** 和 **Bonsai 27B** 在同一批 50 题上的表现，再逐题比较新旧结果。

### 8.3 结果

| Judge | 原版 Prompt | 校准版 Prompt | 变化 |
|---|---|---|---|
| DeepSeek-R1 14B | 13/15（误判 2） | 13/15（误判 2） | **持平**（误判的题换了） |
| Bonsai 27B | 11/15（误判 4） | 12/15（误判 3） | +1 |

逐题变化细节：

| 题号 | 变化 | 解读 |
|---|---|---|
| R1-14B `#27` | B → A | ✅ **修正**：核心答案"初级卵母细胞"正确，过程描述有误但不构成矛盾 |
| R1-14B `#5` | A → B | ❌ **校准引入新错误**：答对"先驱者10号"却因补充说明被判 B |
| Bonsai `#42` | C → B | ✅ **修正**：C 滥用被治好，正确识别出"答错" |
| Bonsai `#47` | B → C | ✅ **修正**：拒答被正确归为 C |
| Bonsai `#2` | A → A | ❌ 仍误判：答"社会民主主义"被当成"社会主义" |

### 8.4 结果解读

1. **DeepSeek-R1 14B 的总误判数没有变化。** 校准版修正了 `#27`，但在 `#5` 引入了新的误判。根据这组样本，继续强化边界规则并未改善其总体结果，因此正式评测仍使用官方模板，以保持与官方评分口径一致。
2. **Bonsai 27B 在部分边界案例上有所改善。** 校准版将 `#42` 从 C 改为 B，也将 `#47` 的拒答正确判为 C；不过，`#2` 仍然误判，说明仅靠这次 Prompt 修改无法消除它的偏松倾向。因此，Bonsai 仍作为对照 Judge，而不用于正式全量评分。

这次实验没有改善正式 Judge 的总体表现，但帮助确认了提示词调整的效果有限，也为保留官方模板提供了依据。

## 9 全量评测结果

**规模**：3 个目标模型 × 3000 题 = **9000 条判定**，全部由同一个 Judge 完成。

### 9.1 总体指标

| # | 模型 | 架构范式 | A 正确 | B 错误 | C 未尝试 | Correct% | Acc(已作答) | **F1** | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **DeepSeek-V2 16B** | MoE | 1110 | 1464 | 426 | 37.00% | 43.12% | **39.83%** | ±1.7 |
| 2 | Qwen2.5 7B Instruct | Dense/Instruct | 980 | 1796 | 224 | 32.67% | 35.30% | 33.93% | ±1.7 |
| 3 | DeepSeek-R1 7B | Reasoning | 301 | 2032 | 667 | 10.03% | 12.90% | 11.29% | ±1.1 |

> **指标口径**（官方 ChineseSimpleQA 定义）：
> `Correct% = A / 总数`；`Acc(已作答) = A / (A+B)`；
> `F1 = 2 × Acc(已作答) × Correct% / (Acc(已作答) + Correct%)`。
> 以 **F1** 为主指标，因为它同时惩罚"答错"和"大量拒答"。

### 9.2 分类别表现（Correct%）

| 类别 | n | DeepSeek-V2 16B | Qwen2.5 7B | DeepSeek-R1 7B |
|---|---|---|---|---|
| 中华文化 | 326 | **44.79%** | 28.83% | 5.21% |
| 人文与社会科学 | 609 | **41.54%** | 38.26% | 8.21% |
| 自然与自然科学 | 530 | 33.40% | **36.04%** | 16.42% |
| 工程、技术与应用科学 | 481 | **40.12%** | 38.25% | 16.84% |
| 社会 | 453 | **35.10%** | 34.44% | 9.49% |
| 生活、艺术与文化 | 601 | **30.28%** | 20.30% | 3.83% |

### 9.3 结果中的几个现象

**① 参数规模不能单独解释结果。**
MoE 架构的 DeepSeek-V2 16B 在总体 F1 和各类别表现上领先于 Dense 的 Qwen2.5 7B。两者在**中华文化**类别的差距最大，为 44.79% 对 28.83%（约 16 个百分点）。这表明，在中文文化知识这一类别中，单看参数量不足以解释模型间的差距，架构和训练数据的差异也值得考虑。

**② DeepSeek-R1 7B 的拒答比例较高。**
DeepSeek-R1 7B 有 **22.2%** 的题目被判为拒答（667 题），而 Qwen2.5 7B 为 **7.5%**（224 题），前者约为后者的 3 倍。检查 C 类输出时，可以看到不少回答采用类似下面的拒答表述：

```text
对不起，我还没有学会回答这个问题。如果你有其他问题，我非常乐意为你提供帮助。
```

ChineseSimpleQA 主要评估事实型短答案，未作答会降低 Correct% 和 F1。因而，这里的低分至少部分反映了模型较保守的作答策略；仅凭这组结果，还不能把差距完全归因于知识覆盖面。实际使用时，也需要同时考虑模型答对的比例和拒答频率。

**③ 三个模型的判定存在较大差异。**
三个模型的 A/B/C 判定一致率最低为 **47.5%**：

| 一致率 | R1 7B | V2 16B | Qwen 7B |
|---|---|---|---|
| DeepSeek-R1 7B | 100% | 47.53% | 52.90% |
| DeepSeek-V2 16B | 47.53% | 100% | 61.97% |
| Qwen2.5 7B | 52.90% | 61.97% | 100% |

各模型单独答对、而另外两个模型没有答对的题目数量如下：

| 模型 | 独有答对 |
|---|---|
| DeepSeek-V2 16B | **409 题** |
| Qwen2.5 7B | 268 题 |
| DeepSeek-R1 7B | 47 题 |

**④ 三个模型同时答对的题目较少。**
按答对模型数量统计，分布如下：

| 答对模型数 | 题数 | 占比 |
|---|---|---|
| 0 / 3（三个全错） | **1526** | **50.87%** |
| 1 / 3 | 724 | 24.13% |
| 2 / 3 | 583 | 19.43% |
| 3 / 3（三个全对） | 167 | 5.57% |

三个模型在 1,526 道题上均未答对，占总题数的 50.87%；三者都答对的题目只有 167 道，占 5.57%。在本次实验所用的本地中小规模模型中，ChineseSimpleQA 的区分难度较高。

## 10 结果检查与局限

除汇总分数外，本项目还检查了数据完整性、输出截断、上下文使用情况、统计不确定性和批次一致性，结果如下：

| 检查项 | 做法 | 结果 |
|---|---|---|
| **数据完整性** | 核对三个模型的预测与判定的 id 集合是否与数据集完全一致 | ✅ 3000/3000，无缺题、无重复 |
| **输出截断** | 统计预测输出长度是否触顶 `max_tokens` | ✅ 最长 1219 字符，远未触顶 |
| **上下文溢出** | 从 llama-server 日志检查 `truncated` 标志 | ✅ 132 次请求全部 `truncated = 0` |
| **单题上下文占用** | 计算 Judge 输入 + 生成的最坏占用 | ✅ 典型 32%，最坏 52%，余量充足 |
| **统计不确定性** | 用 Wilson 区间给出 95% 置信区间 | ✅ 约 ±1~2 个百分点 |
| **批次一致性** | 隔离 smoke test 文件，避免 id 去重导致混批 | ✅ 全量由同一次运行产生 |

### 10.1 局限

> ⚠️ **Judge 仍存在误差。** 在人工核验的 15 道样本中，DeepSeek-R1 14B 的准确率约为 **87%**。因此，模型间的相对排序可用于本次同条件比较，但绝对分数仍可能受到 Judge 误判影响；不同 Judge 得出的绝对分数也不应直接混用。
>
> ⚠️ **本次分数不宜与其他报告直接横向比较。** 本地 7B–16B 模型在本次测试中的 F1 约为 11%–40%。其他报告可能使用不同的模型、Judge 或评测设置，分数差异不一定只来自目标模型本身。

## 11 小结

本项目的主要产出包括三个目标模型的评测结果，以及一组用于检查评分可靠性的记录：

1. 在正式比较模型前，先用人工核验样本检查 Judge 的误判类型和偏差方向。
2. 对 Prompt 的调整逐题比较，并记录误判数和发生变化的具体题目，而不只报告总体比例。
3. 将人工判定与外部事实来源进行核对，并把核实来源一并记录。
4. 如实报告校准实验的结果。校准版 Prompt 没有减少 DeepSeek-R1 14B 的总体误判数，因此全量评测继续使用官方模板。
5. 说明 Judge 在核验样本上的表现及其可能对分数造成的影响。
6. 保留 9,000 条预测和判定记录，方便逐题复查。

---

## 附录 A 仓库结构

```text
ChineseSimpleQA-Local-Eval/
├── run_local_eval.py            # 主流水线（predict / judge / aggregate）
├── run_judge_calibration.py     # Judge 校准实验
├── analyze_calibration.py       # 校准结果分析（含人工核验真值表）
├── make_report.py               # HTML 报告生成（Plotly 交互图表）
├── verify_readme.py             # 本文件数字与原始数据的一致性核验
├── config_local.yaml            # 评测配置
├── requirements.txt             # 运行依赖
├── LICENSE                      # MIT + 第三方内容说明
├── data/
│   └── chinese_simpleqa.jsonl   # 数据集（3000 题）
├── docs/
│   └── index.html               # GitHub Pages 入口（make_report.py 同时产出）
└── results/
    ├── predictions/             # 3 个目标模型的 9000 条预测
    ├── reviews/                 # 9000 条 Judge 判定（逐条含原始输出）
    ├── calibration/             # 50 题校准实验的对照结果
    ├── leaderboard__judge_deepseek-r1_14b__official.csv
    └── report.html              # 完整评测报告（8 张交互图表）
```

## 附录 B 复现命令

```bash
# 环境
conda activate chinese_simpleqa
set OLLAMA_CONTEXT_LENGTH=8192     # 必须在启动 Ollama 前设置

# 1) 目标模型预测（每个模型只加载一次）
python run_local_eval.py --phase predict

# 2) Judge 裁定
python run_local_eval.py --phase judge

# 3) 聚合指标
python run_local_eval.py --phase aggregate

# 4) 生成 HTML 报告
python make_report.py

# 校准实验（可选）
python run_judge_calibration.py --judge deepseek-r1:14b MichelRosselli/bonsai-27b
python analyze_calibration.py
```

## 附录 C 技术栈

`Python 3.11` · `Ollama` · `OpenAI Python SDK` · `pandas` · `Plotly` · `PyYAML` · `tqdm`

---

## 致谢与引用

本项目的**数据集与官方评分逻辑**来自开源项目 **ChineseSimpleQA**：

> **[LivingFutureLab/ChineseSimpleQA](https://github.com/LivingFutureLab/ChineseSimpleQA)**

引用内容与用途说明：

| 内容 | 用途 | 本项目的处理 |
|---|---|---|
| `data/chinese_simpleqa.jsonl`（3000 题） | 评测题库 | 直接使用 |
| `chinese_simpleqa_eval.py` 中的 A/B/C 评分定义与 `GRADER_TEMPLATE` | 官方评分口径 | 在本地运行器中独立复现，**未修改上游文件** |
| `common.py`、`simpleqa_eval.py`、`judge/`、`sampler/` | 上游工具链 | 保留未改（仅 `common.py` 有一处本地并发默认值调整，见下） |

本项目在此基础上**新增**的部分（均为原创）：

- `run_local_eval.py` —— 面向 Ollama 的本地两阶段流水线（含断点续跑、显存卸载、上下文控制）
- `run_judge_calibration.py` + `analyze_calibration.py` —— Judge 校准实验与分析框架
- 校准版 Judge Prompt（9 条边界规则 + B/C 关键区分段落）
- 4 个 Judge 的偏差对比与 15 题人工核验真值表
- `make_report.py` —— HTML 报告生成器

> **说明**：本地 `common.py` 相对上游有一处改动——`map_with_progress` 的默认
> `num_threads` 从 `10` 调整为 `1`（本机配置下并发会带来压力）。
> 该文件不被本地流水线引用，因此**不影响本项目任何评测结果**；
> 仅在上游 `chinese_simpleqa_eval.py` 被直接运行时才起作用。

感谢 [LivingFutureLab](https://github.com/LivingFutureLab) 提供的开源数据集与评测基线。

## 许可

本项目原创代码以 **MIT License** 发布。
上游数据集与代码的版权归其原作者所有，遵循其原始许可（MIT License, Copyright (c) 2024 OpenAI）。
