# ChineseSimpleQA 本地大模型评测实践

> **一个完全本地的中文事实问答评测项目：从"比谁分高"到"确认分数可不可信"**
>
> 在单张 RTX 5080 16GB 上，用 Ollama 部署 3 个不同架构的中文大模型，
> 固定同一套经过人工校准的本地 Judge，完成 3000 题 × 3 模型的统一评测，
> 并完整保留预测与判定原始数据，使实验可复现、可复核、可更换 Judge。

---

## English Abstract

This project builds a fully local evaluation pipeline for **ChineseSimpleQA** (a 3,000-question
Chinese factual QA benchmark) on a single RTX 5080 16GB machine, eliminating network variability
from the measurement.

Three target models spanning distinct architectures — **Qwen2.5 7B Instruct** (Dense/Instruct),
**DeepSeek-R1 7B** (Reasoning), and **DeepSeek-V2 16B** (MoE) — are evaluated against an identical,
human-calibrated local judge (**DeepSeek-R1 14B**), producing 9,000 judgments.

The central finding is methodological rather than a simple leaderboard: **the judge's own error mode
matters more than the accuracy it reports.** Candidate judges were benchmarked on a shared 50-question
set, their failure directions (over-strict / over-lenient / class-C abuse) were identified, 15 disputed
items were manually adjudicated with 5 facts independently web-verified, and a boundary-rule-hardened
prompt variant was designed and A/B tested. That experiment showed the prompt was *not* the bottleneck
for the selected judge — an honest negative result that informed the final design.

**Results (F1):** DeepSeek-V2 16B **39.8%** > Qwen2.5 7B **33.9%** > DeepSeek-R1 7B **11.3%**.
Notably, 50.9% of questions were missed by all three models, and the lowest pairwise
judgment-agreement was only 47.5% — evidence of strong knowledge complementarity despite similar scale.

---

## 1 上游项目本地化：把官方评测搬到本机

本项目的数据集与评分口径来自开源项目 **[LivingFutureLab/ChineseSimpleQA](https://github.com/LivingFutureLab/ChineseSimpleQA)**。
但在本机真正跑起来之前，上游代码有若干处**无法直接复用**。这一节记录逐项定位与处置过程——
它本身也是本项目工作量的一部分。

### 2.1 上游提供的三条评测路径，都依赖外部条件

上游 README 给出三种评测方式：

| 路径 | 依赖 | 本地化障碍 |
|---|---|---|
| `simple-evals` 框架（`python -m simple-evals.demo`） | OpenAI 云端 API | 需联网 + API Key，与本项目「排除网络变量」的目标冲突 |
| 独立脚本 `scripts/chinese_simpleqa_easy.py`（运行入口 `judge/chinese_simpleqa_easy.py`） | 需在源码里硬编码 `OPENAI_API_KEY` / `OPENAI_BASE_URL` | 评测逻辑与云端客户端耦合 |
| OpenCompass 框架 | `git clone open-compass` + 按其 config 配置模型 | 框架重、模型接入方式受限，不适合 Ollama 直连 |

三条路径的共同点是：**评测程序、推理后端、数据格式三者被绑定在一起**，
并不存在一个「自带模型回答」的本地入口。
所以本项目的做法是：**只复用数据集与 A/B/C 评分口径，评测执行链路自己重写**。

### 2.2 五处真实的「水土不服」（逐项查证）

以下问题都是逐行读上游代码定位出来的，不是猜测。

**① 数据格式不匹配** —— 上游读 CSV，本项目用 JSONL

```python
# chinese_simpleqa_eval.py:99
df = pandas.read_csv('chinese_simpleqa.csv')
```

- **硬编码相对路径**：依赖「当前工作目录下正好有个叫 `chinese_simpleqa.csv` 的文件」
- **格式不一致**：本项目使用的是 `data/chinese_simpleqa.jsonl`

**② 字段名不匹配（最隐蔽，会让评测「跑得通但全错」）**

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

**数据里根本没有 `problem` 字段。** 而 `row.get("problem", "")` 带了默认值，
于是它**不会报错，只会安静地取到空字符串** —— 结果是拿空问题去问模型、
再拿空回答去评分。**评测会正常跑完，并输出一份完全无意义的分数。**

> 这是整个本地化过程中最值得记录的一处：**它不崩、不报错、不告警，只是安静地给出错误结论。**
> 重写时把字段名统一为 `question`，并对每条记录做 id 与数据集的集合一致性校验
> （见第 10 节），从机制上堵住这类「静默错误」。

**③ 相对导入与包结构**

```python
# chinese_simpleqa_eval.py:5-6
from . import common
from .types_local import Eval, EvalResult, SamplerBase, SingleEvalResult
```

`from . import ...` 是**包内相对导入**，意味着该文件必须作为某个包的一部分被导入，
不能直接 `python chinese_simpleqa_eval.py` 运行。
本地重写时改为同级模块直接 `import`，不依赖包结构。

**④ 多余的重量级依赖**

```python
# chinese_simpleqa_eval.py:3-4
import blobfile as bf
import pandas
```

`blobfile` 是面向云端对象存储的库，本地评测完全用不到——实测本环境**并未安装**它
（`ImportError`）。`pandas` 对这个规模的数据也非必需。
本地重写只依赖 `openai`（指向 Ollama 的 OpenAI 兼容端点）、`pyyaml`、`requests`、`tqdm`，
HTML 报告额外用 `plotly`。

**⑤ 评分正则的解析缺陷（影响判定正确性）**

```python
# chinese_simpleqa_eval.py:121
match = re.search(r"(A|B|C)", grading_response)
return match.group(0) if match else "C"
```

两个问题：

1. **取的是第一个匹配，而不是最后一个。** 当 Judge 的回复里含推理过程时，
   会取到推理文本里先出现的字母，而不是最终结论。
2. **字母会误匹配英文单词中的字符**（如 "Correct" 里的 `C`）。

而本项目选定的 Judge 是思考模型 **DeepSeek-R1 14B**，其输出天然包含大段推理文本，
这个正则几乎必然出错。本地重写为：

```python
# 先剥离 <think>…</think> 思考块（含未闭合的截断情形）
# 再取最后一个独立的 \b[ABC]\b 匹配
# 仅当剥离后无匹配时，才回退到全文匹配；最终兜底返回 C
```

这是本地化过程中**唯一一处触及「评分正确性」而非「能否跑通」的改动**。

### 2.3 本地化的处置原则

面对上述问题，有两条路：**改上游代码**，或者**只复用口径、重写执行链路**。

选择后者，原则是：

| 原则 | 具体做法 |
|---|---|
| 上游文件保持可对照 | `chinese_simpleqa_eval.py`、`simpleqa_eval.py`、`judge/`、`sampler/` 均不改动，作为口径的权威参照 |
| 评分定义以官方为准 | A/B/C 的定义、数值精度规则、异体字规则、F1 计算公式，全部照搬官方 `GRADER_TEMPLATE` 与官方聚合逻辑 |
| 只在必要处重写 | 数据加载（CSV→JSONL）、字段映射、模型调用（云端→Ollama）、A/B/C 解析、断点续跑、结果落盘 |
| 唯一的上游改动显式声明 | `common.py` 的 `map_with_progress` 默认 `num_threads` 由 `10` 调整为 `1`（本机并发压力）。该文件**不被本地流水线引用**，因此不影响本项目任何评测结果 |

**为什么坚持不改上游**：一套评测要能被别人复核，就必须能分清
「哪些是官方定义、哪些是我的实现」。上游文件原样保留，
任何人都可以把它和重写版逐行对照，确认 A/B/C 口径没有被偷改。

### 2.4 本地化后的运行链路

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

整条链路只依赖本机 Ollama 的 OpenAI 兼容端点，**运行时不需要任何外网访问**。

## 2 项目缘起

最初的动机很朴素：想比较几个本地可部署的中文模型，在事实问答任务上到底差多少。

第一次尝试走的是在线 API 路线，很快遇到了问题：**网络稳定性、代理、请求限流都会污染测量结果**。
同一条 prompt 在不同网络条件下重试，得到的分数可能漂移好几个百分点——这时候你测的到底是模型能力，
还是网络质量？

于是决定把整条链路搬到本机：**模型推理（Ollama）+ Judge + 数据集 + 评测程序**全部本地化。
网络因素被彻底排除后，剩下的波动就只能来自模型本身和评测设计——这才是可控的实验。

项目真正的转折点出现在 Judge 环节。最初的方案是用一个本地模型当裁判，后来人工抽查发现
**这个裁判自己的判错率相当高**。从那一刻起，项目的重心从"比较模型分数"转向了
**"先确认这把尺子准不准"**。

## 3 硬件与环境约束

| 项 | 配置 |
|---|---|
| GPU | NVIDIA RTX 5080，16 GB VRAM |
| CPU | AMD Ryzen 7 5700X3D |
| RAM | 32 GB DDR4-3200 |
| OS | Windows 10 |
| 推理后端 | Ollama（OpenAI 兼容 API，`http://localhost:11434/v1`） |

16 GB 显存决定了模型选择的上限，也直接塑造了整个架构：**任何时刻只能有一个模型驻留显存**。
这个约束不是缺陷，反而逼出了一套更干净的流水线设计。

### 2.1 上下文长度：一个被低估的陷阱

项目早期踩过一个坑：Ollama 默认上下文窗口很大（部分模型标签写着 131072），
在 16 GB 卡上会导致 `llama-server` 占用飙升、部分层被挤到 CPU：

```text
deepseek-r1:14b   35 GB   60% CPU / 40% GPU   131072 context
```

推理速度随之崩掉。把上下文固定到 **8192** 之后：

```text
100% GPU
```

**关键认识：评测任务里每一道题都是独立请求，3000 道题不会累加进同一个上下文。**
所以 128K 的窗口对一个"单题输入只有一两千字符"的 Judge 来说完全是浪费。
这道优化后来直接决定了整个项目的可行性。

实测确认（llama-server 日志）：

```text
-c 8192 -np 1 ... offloaded 29/29 layers to GPU
llama_kv_cache: size = 448.00 MiB (8192 cells, 28 layers, 1/1 seqs)
```

各模型的原生窗口上限也一并查清（决定了上下文能开多大）：

| 模型 | 原生上限 `n_ctx_train` | 8K 下实测显存 |
|---|---|---|
| Qwen2.5 7B Instruct | 32,768 | 4,896 MiB |
| DeepSeek-R1 7B | 32,768 | ~4.9 GB |
| DeepSeek-R1 14B（Judge） | 131,072 | **9,922 MiB** |
| Bonsai 27B（对照 Judge） | 262,144 | 4,367 MiB |

## 4 评测架构

### 3.1 两阶段流水线

```text
Phase 1  Predict（逐模型串行）
  加载 Target → 跑完 3000 题 → 保存 predictions → 卸载 → 下一个 Target

Phase 2  Judge（单独加载 Judge）
  加载 Judge → 评该 Target 的全部 predictions → 保存 reviews → 卸载

Phase 3  Aggregate
  读 reviews → 计算指标 → 输出 leaderboard CSV + HTML 报告
```

### 3.2 四条设计约束

1. **模型只加载一次。** 绝对不做"每题加载→回答→卸载"——3000 题那样跑会浪费掉绝大部分时间在权重加载上。
2. **并发 = 1。** 16 GB 显存下串行调用，同时保证结果一致性。
3. **Target 与 Judge 绝不同时驻留显存。**
4. **逐条 JSONL 追加 + 立即 flush。** 任何时刻中断，已完成的题都不会丢。

### 3.3 断点续跑的语义

续跑按 `id` 去重：启动时读取已有 JSONL，已完成的题目直接跳过。
这让长任务变得可以随时中断——关掉电脑、第二天接着跑，已判定的题不会重跑。

> ⚠️ **这里有一个容易忽略的副作用**：正因为是"按 id 跳过"，如果目录里残留着
> 早期小规模 smoke test 的判定文件，正式全量运行时那批题会被**静默跳过**，
> 最终结果是"旧批次 + 新批次"的混合体，破坏批次一致性。
> 本项目在正式运行前把 50 题的 smoke test 文件改名隔离（`.smoketest`），
> 确保 3000 题全部由同一次运行产生。**这类问题不会报错，只会悄悄污染数据。**

**输出命名**（不同 Judge / 不同 Prompt 的结果并存不覆盖）：

```text
predictions/{target}.jsonl
reviews/{target}__judged_by__{judge}__{prompt_version}.jsonl
leaderboard__judge_{judge}__{prompt_version}.csv
```

## 5 Judge 提示词工程 ⭐

这是整个项目中**最能体现评测设计能力**的部分。

### 4.1 为什么需要自定义 Judge Prompt

官方 `chinese_simpleqa_eval.py` 的评分模板定义了 A/B/C 三档：

- **A = 正确**：包含参考答案要点，且不含矛盾信息
- **B = 错误**：包含与参考答案矛盾的事实陈述
- **C = 未尝试**：没有给出答案，也没有矛盾陈述

但直接用官方模板让本地 7B~27B 模型当裁判，会出现系统性问题：

| 候选 Judge | 偏差方向 | 典型误判 |
|---|---|---|
| Qwen2.5 7B | **偏严** | 答对但补充了信息 → 误判 B；拒答 → 误判 B 而非 C |
| DeepSeek-R1 7B | **偏松** | 答错 → 误判 A |
| DeepSeek-R1 14B | 偏严（较轻） | 对"已答对但补充无关信息"的题判 B |
| Bonsai 27B | **偏松 + C 滥用** | 答错 → 误判 C；提到关键词即判 A |

**注意偏差方向的意义完全不同**：偏严会低估模型分数，偏松会虚高分数。
如果一个 Judge 把所有"答错"都判成"未尝试"，那么一个胆小拒答的模型反而会得到漂亮分数——
**这不是评测，这是被评测对象牵着走。**

### 4.2 校准版 Prompt 的设计

为此设计了一份边界规则显式化的校准版模板，核心改动：

1. **开头用英文显式声明 A/B/C 定义**（对 Bonsai 这类模型英文指令更稳定）
2. **9 条 CRITICAL BOUNDARY RULES** 把边界写死，例如：
   - 规则 3：正确答案 + 与核心答案矛盾的事实错误 → B
   - 规则 4：**答错（模型尝试了但答错）→ B，不是 C**
   - 规则 6：**否认题目前提、但仍提到关键词 → 不算 A**
   - 规则 7：**人名异体字 / 不同音译 → A**（如 喻浩 / 喻皓）
   - 规则 8：数值精度规则（参考值 3518.17，答 3518 / 3518.1 / 3518.17 均为 A；答 3520 为 B；答"大约 3500"为 C）
3. **单独一段 `KEY DISTINCTION: B vs C`**，明确"答错 ≠ 未尝试"
4. **结尾强调只输出单个字母**，降低解析成本

其中第 2 条里的规则 3/4/6/7 都是直接从**人工核验过的误判案例**反向提炼出来的——
不是凭空设计的规则，而是"先发现错误模式，再写规则约束它"。这正是数据标注规范的实际工作方式。

### 4.3 实现细节：模板单点定义

为了避免两份 Prompt 副本漂移，校准版模板只在 `run_local_eval.py` 中定义一次，
校准脚本通过 import 复用：

```python
from run_local_eval import CALIBRATED_GRADER_TEMPLATE, CALIBRATED_JUDGE_SYSTEM_MESSAGE
```

改 Prompt 只需改一处，两个脚本行为自动一致。

## 6 人工核验闭环 ⭐

**Judge 说 A/B/C 不算数——必须有人去核对。**

### 5.1 方法

从 50 题 smoke test 中挑出 **4 个 Judge 判定分歧的 15 道题**，逐题人工判定真值。
判定依据严格对齐官方口径：A = 含参考答案要点且无矛盾；B = 含矛盾陈述或给出不同实体/数值；C = 未答且无矛盾。

### 5.2 一个必须诚实说明的过程

**我最初的判断有 3 题是错的，是联网查证之后才纠正的。**

| 题号 | 我的初判 | 核实后 | 依据 |
|---|---|---|---|
| #16 | B | B ✓ | 卢梭《论人类不平等的起源和基础》确是 **1755 年** 4 月初版于阿姆斯特丹；模型答"1754年"与参考答案矛盾 |
| #20 | B | B ✓ | 1954 年被伊朗当局审判的是**伊朗人民党（图德党）**；模型给出"伊朗民族解放阵线"这一不同组织 |
| #42 | — | B | 书目数据库存储的是**二次文献**；模型答"元数据"，给出了不同概念 |
| #44 | B | B ✓ | 金斗瓮在广东、香港俗称**金塔**；模型答"猪笼草" |
| #50 | B | B ✓ | 释行真确为少林寺**第三十二代**嫡传弟子；模型称其"并非少林寺嫡传弟子" |

核实来源：[卢梭著作（百度百科）](https://baike.baidu.com/item/%E8%AE%BA%E4%BA%BA%E7%B1%BB%E4%B8%8D%E5%B9%B3%E7%AD%89%E7%9A%84%E8%B5%B7%E6%BA%90%E5%92%8C%E5%9F%BA%E7%A1%80/5880820) ·
[金斗瓮（维基百科）](https://zh.wikipedia.org/zh-hans/%E9%87%91%E6%96%97%E7%94%95) ·
[释行真（百度百科）](https://baike.baidu.com/item/%E9%87%8A%E8%A1%8C%E7%9C%9F/3793721) ·
[伊朗人民党（维基百科）](https://zh.wikipedia.org/zh-hans/%E4%BC%8A%E6%9C%97%E4%BA%BA%E6%B0%91%E5%85%9A)

**这件事的意义大于那几道题本身**：连人工标注者的判断都需要被外部事实源复核。
一套只依赖"我觉得"的评测流程，本质上和被测模型一样不可靠。
最终 15 题的真值表固化在 `analyze_calibration.py` 的 `HUMAN_GROUND_TRUTH` 中，可随时复现与增改。

## 7 四个 Judge 的横向对比

用**同一批 50 题、同一个目标模型（Qwen2.5 7B）**，让 4 个候选 Judge 分别判定：

| Judge | A | B | C | 报告的 Correct% | 人工核验误判 |
|---|---|---|---|---|---|
| Qwen2.5 7B | 13 | 37 | 0 | 26.0% | 偏严，A 明显偏低 |
| DeepSeek-R1 7B | 20 | 29 | 1 | 40.0% | 偏松，答错判成 A |
| **DeepSeek-R1 14B** | 16 | 31 | 3 | 32.0% | **2 / 15（13.3%）** |
| Bonsai 27B | 17 | 28 | 5 | 34.0% | **4 / 15（26.7%）** |

注意 `Qwen2.5 7B` 的 C 是 **0** ——它在 50 题里一次都没判过"未尝试"，
说明它对"拒答"这个类别的识别基本失效。而 R1-7B 报出的 40% 是三家里最高的，
但人工核对发现这个数字被"答错判成 A"的偏差抬高了。

> **最重要的结论不是谁报的百分比最高，而是：Judge 自身的误判模式，比它报告的分数更重要。**

### 6.1 速度也是决策变量

| Judge | 单题耗时 | 9000 条预计 |
|---|---|---|
| Qwen2.5 7B | ~0.6 s | ~1.5 h |
| DeepSeek-R1 7B | ~1.7 s | ~4.3 h |
| **DeepSeek-R1 14B** | **~3.2 s** | **~8 h** |
| Bonsai 27B | ~10 s | ~25 h |

最终选择 **DeepSeek-R1 14B** 的三条理由：

1. **准确率**：人工核验题上误判 2/15，优于 Bonsai 的 4/15（原版）
2. **速度**：比 Bonsai **快约 3 倍**（9000 条：约 8 小时 vs 约 25 小时）
3. **偏差方向更安全**：它是"偏严"（可能低估分数），而 Bonsai 是"偏松"（虚高分数）。
   在评测里，**虚高比低估更危险**——它会把不存在的能力写进结论。

## 8 校准实验：一个诚实的负结果

### 7.1 实验设计

**要回答的问题**：剩下那些误判，是 **Prompt 表述模糊**导致的，还是 **模型能力上限**导致的？

这个问题直接决定下一步该做什么：
- 如果是 Prompt 问题 → 继续优化 Prompt
- 如果是能力上限 → 换 Judge，或接受并记录偏差

### 7.2 做法

用第 4 节设计的校准版 Prompt，把 **DeepSeek-R1 14B** 和 **Bonsai 27B** 在同一批 50 题上重跑一遍，
再与各自的原版结果逐题对比。

### 7.3 结果

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

### 7.4 结论

1. **对 DeepSeek-R1 14B，校准版 Prompt 净收益为 0。**
   它修好了 `#27`，却又弄坏了 `#5`。这说明其残留误判来自
   **模型能力上限，而不是 Prompt 表述模糊**。
   因此正式评测**采用官方模板**，保持与官方口径一致。
2. **校准版 Prompt 确实治好了 Bonsai 的"C 滥用 / 把拒答判成 B"**（`#42`、`#47`），
   证明**提示词修改是有效的**——但它的偏松问题（`#2`）仍然存在，那是模型级问题，Prompt 解决不了。
   所以 Bonsai 只保留为对照 Judge。

> 这个"没提升"的结果本身就是有价值的产出：
> **它把"继续调 Prompt"这条死路提前排除掉了，避免了在正式评测上浪费时间。**
> 评测设计的一部分工作，就是证明某条路走不通。

## 9 全量评测结果

**规模**：3 个目标模型 × 3000 题 = **9000 条判定**，全部由同一个 Judge 完成。

### 8.1 总体指标

| # | 模型 | 架构范式 | A 正确 | B 错误 | C 未尝试 | Correct% | Acc(已作答) | **F1** | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **DeepSeek-V2 16B** | MoE | 1110 | 1464 | 426 | 37.00% | 43.12% | **39.83%** | ±1.7 |
| 2 | Qwen2.5 7B Instruct | Dense/Instruct | 980 | 1796 | 224 | 32.67% | 35.30% | 33.93% | ±1.7 |
| 3 | DeepSeek-R1 7B | Reasoning | 301 | 2032 | 667 | 10.03% | 12.90% | 11.29% | ±1.1 |

> **指标口径**（官方 ChineseSimpleQA 定义）：
> `Correct% = A / 总数`；`Acc(已作答) = A / (A+B)`；
> `F1 = 2 × Acc(已作答) × Correct% / (Acc(已作答) + Correct%)`。
> 以 **F1** 为主指标，因为它同时惩罚"答错"和"大量拒答"。

### 8.2 分类别表现（Correct%）

| 类别 | n | DeepSeek-V2 16B | Qwen2.5 7B | DeepSeek-R1 7B |
|---|---|---|---|---|
| 中华文化 | 326 | **44.79%** | 28.83% | 5.21% |
| 人文与社会科学 | 609 | **41.54%** | 38.26% | 8.21% |
| 自然与自然科学 | 530 | 33.40% | **36.04%** | 16.42% |
| 工程、技术与应用科学 | 481 | **40.12%** | 38.25% | 16.84% |
| 社会 | 453 | **35.10%** | 34.44% | 9.49% |
| 生活、艺术与文化 | 601 | **30.28%** | 20.30% | 3.83% |

### 8.3 四个值得注意的发现

**① 参数量不是唯一变量。**
DeepSeek-V2 16B 是 MoE 架构（激活参数远小于 16B），却全面领先 Dense 的 Qwen2.5 7B。
差距在**中华文化**类别最大：44.79% vs 28.83%（**+16 个百分点**）。
这说明在中文文化知识这个维度上，架构与训练数据的差异比单纯参数量更重要。

**② 思考模型在这个任务上明显吃亏——但原因不是"更笨"。**
DeepSeek-R1 7B 的拒答率高达 **22.2%**（667 题），而 Qwen2.5 7B 只有 **7.5%**（224 题），**相差 3 倍**。
翻看它的 C 类输出，大量是：

```text
对不起，我还没有学会回答这个问题。如果你有其他问题，我非常乐意为你提供帮助。
```

而 ChineseSimpleQA 是**事实型短答案**任务——拒答就直接丢分。
**它的知识广度未必输，输在作答策略过于保守。**
这个发现对实际应用有直接意义：评估一个模型能不能用，不能只看它在"敢答的题"上答得多准。

**③ 模型间知识高度互补。**
三个模型 A/B/C 判定的一致率最低只有 **47.5%**：

| 一致率 | R1 7B | V2 16B | Qwen 7B |
|---|---|---|---|
| DeepSeek-R1 7B | 100% | 47.53% | 52.90% |
| DeepSeek-V2 16B | 47.53% | 100% | 61.97% |
| Qwen2.5 7B | 52.90% | 61.97% | 100% |

"只有某一个模型答对"的题目数量：

| 模型 | 独有答对 |
|---|---|
| DeepSeek-V2 16B | **409 题** |
| Qwen2.5 7B | 268 题 |
| DeepSeek-R1 7B | 47 题 |

**④ 题目本身很难。**
答对题数的分布：

| 答对模型数 | 题数 | 占比 |
|---|---|---|
| 0 / 3（三个全错） | **1526** | **50.87%** |
| 1 / 3 | 724 | 24.13% |
| 2 / 3 | 583 | 19.43% |
| 3 / 3（三个全对） | 167 | 5.57% |

**超过一半的题目三个模型全部答错**，只有 5.57% 是三者都会。
ChineseSimpleQA 对本地中小规模模型而言是一个相当困难的基准。

## 10 结果可信度的方法学工作

一个评测报告的分数如果不说明"这些数字可能错在哪"，那它的价值是有限的。
本项目做了以下几项检查：

| 检查项 | 做法 | 结果 |
|---|---|---|
| **数据完整性** | 核对三个模型的预测与判定的 id 集合是否与数据集完全一致 | ✅ 3000/3000，无缺题、无重复 |
| **输出截断** | 统计预测输出长度是否触顶 `max_tokens` | ✅ 最长 1219 字符，远未触顶 |
| **上下文溢出** | 从 llama-server 日志检查 `truncated` 标志 | ✅ 132 次请求全部 `truncated = 0` |
| **单题上下文占用** | 计算 Judge 输入 + 生成的最坏占用 | ✅ 典型 32%，最坏 52%，余量充足 |
| **统计不确定性** | 用 Wilson 区间给出 95% 置信区间 | ✅ 约 ±1~2 个百分点 |
| **批次一致性** | 隔离 smoke test 文件，避免 id 去重导致混批 | ✅ 全量由同一次运行产生 |

### 9.1 必须声明的局限

> ⚠️ **Judge 本身不完美。** 经人工核验，选定的 DeepSeek-R1 14B 在 15 题样本上准确率约 **87%**。
> 因此：
> - **各模型之间的相对排名是可信的**（同一把尺子量所有人）
> - **但绝对分数存在约 ±1~2 个百分点的系统性偏差**
> - **不同 Judge 下的绝对分数不可直接混用**
>
> ⚠️ **绝对分数偏低是正常的。** 这类基准的公开成绩通常来自 GPT-4 级别的闭源模型。
> 本地 7B~16B 模型拿到 30%~40% 属于合理区间，**不应与其他报告的绝对值并列比较**。

**主动写出这些限制，比隐藏它们更能体现评测工作的专业性。**
一个知道"自己的数字什么时候不能信"的人，才是能设计出可信评测的人。

## 11 小结

这个项目表面上产出的是三个模型的分数，实际上建立的是**一套判断"评测结果可不可信"的工作方法**：

1. **先校准尺子，再量东西。** Judge 的偏差方向必须先查清，否则测出来的分数没有意义。
2. **提示词修改要有因果证据。** 不是"我优化了 prompt"，而是"改前误判 4/15，改后 3/15，具体是哪几题变了、为什么变"。
3. **人工标注必须被外部事实源复核。** 我自己最初也判错了 3 题。
4. **负结果要如实记录。** 校准版 Prompt 对选定的 Judge 没有净提升——这个结论避免了后续无效投入。
5. **主动披露局限。** 说明 Judge 准确率约 87%、绝对分数有系统偏差，比给一个漂亮数字更重要。
6. **原始数据必须留全。** 9000 条判定逐条可查，任何人都能复核任意一题的判定依据。

---

## 附录 A 仓库结构

```text
chinese-simpleqa-eval/
├── run_local_eval.py            # 主流水线（predict / judge / aggregate）
├── run_judge_calibration.py     # Judge 校准实验
├── analyze_calibration.py       # 校准结果分析（含人工核验真值表）
├── make_report.py               # HTML 报告生成（Plotly 交互图表）
├── config_local.yaml            # 评测配置
├── data/
│   └── chinese_simpleqa.jsonl   # 数据集（3000 题）
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
