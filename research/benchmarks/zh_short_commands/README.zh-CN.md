# 中文短指令路由

[English](README.md) · [同类基准：中文职场决策](../feishu_zh/README.zh-CN.md) · [Issue #218](https://github.com/NandhaKishorM/laya/issues/218)

18 条冻结的中文清洁机器人语音指令、6 个标签，以及针对仓库自身提示词建议的 7 档消融阶梯。一个 checkpoint、一次运行、逐条决策全部归档——目的是把 [#218](https://github.com/NandhaKishorM/laya/issues/218) 的一次性报告（"加上 criteria、场景描述和结构化 JSON state 之后，中文决策反而更糟"）变成任何人都能复算的工件，并且说清它究竟对哪个原语成立。

**这 18 条是手写夹具，标签策略在任何模型运行之前就已固定。它不是留出测试集，不是独立标注语料，也不是官方评测。** 下面的数字只是一个 checkpoint 在这些夹具上的决策。

## 先做这一步：离线自证

在仓库根目录下，Python 3.10+：

```bash
python research/benchmarks/zh_short_commands/audit.py
python -m unittest discover -s research/benchmarks/zh_short_commands/tests -v
```

不需要下载模型、不需要联网。审计只用标准库，CI 把它放在一个什么都没安装的 job 里运行。测试套件会驱动 `run.py --stub`，因此需要 numpy——任何能跑 Laya 的环境本来就有。

审计会重新校验冻结用例与提示词的哈希、重新构造每一个请求（state、instructions、options、gold）而不是信任记录本身，并仅凭逐条记录复算 accuracy、macro-F1、ECE 和置信度离散度。测试会用 18 种方式篡改归档副本，要求审计全部拒绝；这些测试存在的意义就是证明这个审计真的会失败。

## 阶梯

`choice` 不可能没有 criteria——criteria 的键**就是**它的选项——所以任务 A 的起点比任务 B 高一档。

**任务 A，`choice`：每条指令一个问题，六选一。**

| 档位 | 正确 | accuracy | macro-F1 | ECE | 平均置信 | 置信标准差 |
|---|---:|---:|---:|---:|---:|---:|
| `choice_criteria`（只有六个选项） | 13/18 | 0.7222 | 0.7149 | 0.2659 | 0.8008 | 0.1728 |
| `choice_scenario`（+场景句） | 14/18 | 0.7778 | 0.7942 | 0.1827 | 0.8257 | 0.1725 |
| `choice_json_state`（+结构化 state） | 12/18 | 0.6667 | 0.6817 | 0.2484 | 0.7961 | 0.1238 |

最高频标签只覆盖 5/18，所以多数类基线是 0.2778。

**任务 B，`noul`：每条指令四个独立的 yes/no 问题，共 72 个决策。**

| 档位 | 正确 | accuracy | 平均置信 | 置信标准差 |
|---|---:|---:|---:|---:|
| `noul_plain`（无 criteria，单问题） | 48/72 | 0.6667 | 0.8128 | 0.1786 |
| `noul_criteria` | 33/72 | 0.4583 | 0.8920 | 0.0944 |
| `noul_scenario` | 34/72 | 0.4722 | 0.8944 | 0.0807 |
| `noul_json_state` | 30/72 | 0.4167 | 0.9437 | 0.0418 |

对 72 个问题全答 "true" 得 30/72 = 0.4167，因为 72 个 gold 答案里只有 30 个为真。`noul_json_state` 拿到的正好就是这个分数。

## 阶梯说明了什么

文档建议并没有让 `noul` 路径"更懂中文"，而是让这四个问题不再有区分度。统计每个维度对 18 条指令答 "true" 的条数：

| 档位 | `wants_faster`（真值 4） | `wants_slower`（5） | `wants_stop`（4） | `is_command`（17） |
|---|---|---|---|---|
| `noul_plain` | 9 — acc 0.611 | 14 — 0.389 | 5 — 0.944 | 12 — 0.722 |
| `noul_criteria` | 17 — 0.278 | 17 — 0.333 | **18** — 0.222 | 17 — **1.000** |
| `noul_scenario` | 17 — 0.278 | 17 — 0.333 | 17 — 0.278 | 17 — **1.000** |
| `noul_json_state` | **18** — 0.222 | **18** — 0.278 | **18** — 0.222 | **18** — 0.944 |

一旦加上 criteria，`wants_stop` 对全部 18 条指令都答 "true"——包括那 14 条根本不是停止指令的，且每条的 p(true) ≥ 0.794。到最高档，四个维度对每一条指令都答 "true"。`is_command` 在 criteria 档拿到 1.000，只是因为那里 "true" 本来就是常见答案（17/18）；再叠上 JSON state 后它退回全真基线（0.944）——此时它还会对 今天天气不错 说"是运动指令"，p(true) = 0.938，而 `noul_plain` 给的是 0.005。

所以准确率的下降是答案分布的属性，不是建议真的教给了模型什么中文知识。置信度离散度从另一侧印证同一件事：最高档 72 个置信度全部落在 0.750–0.995，而 `noul_plain` 覆盖 0.504–1.000。

**任务 A 没有这个问题。** 三个 choice 档都没有塌缩：0.7222 → 0.7778 → 0.6667，场景句多对了 1 条，JSON state 少对了 2 条。`choice` 问题不需要回答 "是/否"——criteria 的键就是它的选项，答案无法退化成常数。这正是 [#218](https://github.com/NandhaKishorM/laya/issues/218) 看不到的差别：它测的那套建议对 `noul` 路径有害、对 `choice` 路径无害，两者不能平均成一个"中文准确率"。

## 阶梯改变不了的部分

有 6 个错误在三个 choice 档里完全一致，说明它们是 checkpoint 在这些输入上的属性，而不是提示词格式造成的：

| # | 指令 | Gold | 三档结果 | 置信度 |
|---|---|---|---|---|
| 001 | 快一点 | `faster` | `slower` | 0.652 / 0.690 / 0.703 |
| 003 | 太慢了 | `faster` | `slower` | 0.990 / 0.993 / 0.898 |
| 006 | 太快了 | `slower` | `faster` | 0.795 / 0.901 / 0.767 |
| 016 | 往右边靠一下 | `right` | `left` | 0.395 / 0.399 / 0.556 |
| 010 | 别动了 | `stop` | `none`（3 档中的 2 档） | 0.607 / – / 0.796 |
| 017 | 别太快了 | `slower` | `faster`（仅 JSON state 档） | – / – / 0.556 |

只看速度族：太慢了 意为"太慢"（要更快），太快了 意为"太快"（要减速）——模型选的是形容词所指的方向，而不是整句要求的纠正动作，而且带着 0.90–0.99 的置信度。否定是被处理的：别太快了 和 别动了 至少在一个档里被读对，别太快了 甚至被场景句纠正过来。所以这个模式是针对 太（"过度"）构式反转动作的，不是中文整体的失败。

方向路由也不是整体崩坏：左转、往左一点、向右转 在三个档里全对，唯一的失败（往右边靠一下 → `left`）是归档里置信度最低的一次决策。闲聊会被拒绝：今天天气不错 → `none`，三个档全对。上面每一行都**只是一个夹具**——它们是"下一批 100 条该往哪儿放"的廉价可证伪假设，不是已确立的构式结论。

**两个任务的形状本身就不一致。** 有 6 条指令，`choice_criteria` 与 `noul_plain` 的 `is_command` 互相矛盾：

| # | 指令 | `choice` 判定 | `is_command` p(true) |
|---|---|---|---|
| 001 | 快一点 | `slower` | 0.254 |
| 005 | 慢一点 | `slower` | 0.496 |
| 010 | 别动了 | `none` | 0.681 |
| 013 | 左转 | `left` | 0.115 |
| 014 | 往左一点 | `left` | 0.226 |
| 017 | 别太快了 | `slower` | 0.462 |

产品侧只能选一种工作流并针对它标定，不能同时消费两者。这与[同类飞书基准](../feishu_zh/README.zh-CN.md)对它的 `choice`/`four_noul` 组合得出的结论一致。

## 溯源

由 `run.py` 写入归档的 `config.checkpoint`；缺少权重指纹的归档会被审计直接拒绝。

- Checkpoint：`convaiinnovations/laya` 的 `multilingual/` 子目录，从本地目录读取（`E:/home/laya-models`）。仓库内置副本与独立仓库 `convaiinnovations/laya-multilingual` 目前对权重报告同一个 LFS 摘要，因此这就是已发布的多语言 checkpoint。
- `model.safetensors`：643835514 字节，`sha256 9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204`。
- 本次运行：2026-09-24T07:56:04Z，CPU，float32，`laya 0.3.20`，`torch 2.14.0+cpu`，`transformers 5.17.0`，Python 3.12。
- 打分方式：使用 checkpoint 自带的分桶温度，并按 `Agent` 的方式做同样的钳制（归档里 `unclamped: false`），与 [`research/eval/laya_eval.py`](../../eval/laya_eval.py) 同一条代码路径。没有拟合任何阈值，也没有做微调。
- 七个配置的完整扫描在 CPU 上约 20 秒。

`research/eval/README.md` 对同一个文件记录的是另一个值——`sha256 b99c8bea…`。那个字符串是 Hub 仓库里**LFS 指针**的 git blob id（SHA-1），不是权重的 SHA-256：把指针文本 `version https://git-lfs.github.com/spec/v1\noid sha256:9d628fd9…\nsize 643835514\n` 按 git blob 规则哈希，得到的正是 `b99c8bea239c53f6f6bce734557dc6c403fa6b3e`。同一段里 `rl_agent_config.json` 的 `sha256 00e35f88…` 同样是该 JSON 自身的 blob id。任何人用 `sha256sum` 去核对都会看到不一致，从而误判"checkpoint 变了"。这里只做标记，不静默修改，因为那属于本贡献所不拥有的另一个文件的改动。

## 在自己的机器上运行

如果本地还没有，只下载多语言运行时文件（约 614 MiB）：

```python
from huggingface_hub import snapshot_download
root = snapshot_download(
    "convaiinnovations/laya",
    allow_patterns=["multilingual/*.json", "multilingual/model.safetensors",
                    "multilingual/encoder/*", "multilingual/tokenizer/*"],
)
print(root + "/multilingual")
```

把 `CHECKPOINT` 设成该目录，核对文件，然后跑扫描。`--stub` 用固定伪随机向量代替 checkpoint，可离线跑通整条流水线；`--configs` 只跑子集：

```bash
python research/benchmarks/zh_short_commands/run.py --checkpoint "$CHECKPOINT" \
  --device cpu --out /tmp/zh-short-commands
python research/benchmarks/zh_short_commands/audit.py --run-dir /tmp/zh-short-commands
```

runner 会把用例、`prompts.py` 和 checkpoint 的 `model.safetensors` 哈希一并写入报告，所以来自另一台机器的归档可以与已提交的归档逐字段比对。每次运行请使用新的输出目录。`--unclamped` 复现旧提交扫描使用的原始未钳制温度；上面的归档数字用的是钳制后的值。

## 文件

| 路径 | 用途 |
|---|---|
| `data/cases.jsonl` | 18 条冻结指令：id、文本、family、gold 标签、备注 |
| `data/manifest.json` | 标签/family 计数、阶梯、判定策略，以及两个冻结输入的哈希 |
| `prompts.py` | 七个档位：instructions、六个 criteria、四个 `noul` 维度及其 criteria |
| `run.py` | runner：加载 checkpoint、跑阶梯、写 `report.json` |
| `audit.py`、`tests/` | 离线复算、18 种归档篡改用例，以及与 `laya_eval` 的指标对照 |
| `results/v1/laya-multilingual/report.json` | 归档运行：7 个 report 块和全部 342 条逐例记录 |

归档的三个部分与 `research/eval/laya_eval.py` 输出的三个部分相同——`config`、`report`、`cases`——外加一个 `summary`。`report` 里的每个数字都不需要模型即可复算。

## 局限与署名

- 18 条用例，由贡献者手写，没有独立标注，也没有标注者一致性。family 不平衡（速度 9、停止 4、方向 4、闲聊 1），标签也不平衡（每个 1–5 条），这正是 `noul` 的基率比档位排序更重要的原因。
- 单一 checkpoint、单一语言、单一设备，固定温度、不做采样。这里没有任何结论适用于英文或 typed-decisions checkpoint。
- 本对比复现的是 [#218](https://github.com/NandhaKishorM/laya/issues/218) 的**形态**，不是它的提示词：那份报告的原文从未公开，因此它的约 50% 与上面的 0.4167–0.6667 不能直接比较。
- 这些夹具是回归诊断。不要在上面调参，再把结果当作留出集报告。
- 中文后训练仍是开放的研究问题；这份文件没有解决中文短指令路由，它只是把它测量出来。

由 GaotianJin 贡献，harness、审计与测试借助 AI 协助完成。用例文本为贡献者本人所写。不包含任何用户数据、录音、凭据或模型权重；checkpoint 仍遵循其自身许可。本目录与仓库其余部分一样采用 Apache-2.0 许可。
