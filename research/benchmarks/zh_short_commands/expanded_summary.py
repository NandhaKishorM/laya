"""Make reviewable tables and an English issue draft from audited real results."""
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import expanded_eval as evaluation


def main():
    here = Path(__file__).resolve().parent
    doc = evaluation.audit(here / "expanded_report.json")
    historical = evaluation.read(here / "results/v1/laya-multilingual/report.json")
    metadata = doc["config"]
    lines = ["# 中文短指令行为评测", "",
             "本次为真实模型推理；96 条新增用例由本次助手按固定规则构造，标签在运行前冻结。",
             "贡献者已于 2026-10-13 审核全部用例及标签并确认无需修改。审核在评测后完成；这不是自然流量或独立标注的留出测试集。",
             "六类各 16 条；原 18 条与新增 96 条分别统计，不合并成一个准确率。",
             "相似改写彼此相关，未报告将其视为独立抽样的显著性检验。未做微调、温度拟合或测试集调参。", "",
             "## 环境与溯源", "",
             f"- 代码：`{metadata['git_commit']}`，Laya {metadata['laya']}",
             f"- Python {metadata['python'].split()[0]}，PyTorch {metadata['torch']}，Transformers {metadata['transformers']}",
             f"- {metadata['device']} / FP32，4 个计算线程，1 个 inter-op 线程，seed 13，确定性算法",
             f"- 模型权重 SHA-256：`{metadata['checkpoint']['weight_sha256']}`",
             "- 原提示词不变，七种配置全部运行；每种配置重复两次，记录原始 logits",
             "- 本报告置信度为最大选项概率，与 SDK 的 choice 熵置信度字段不同", "",
             "## 原 18 条基线复现", "",
             "| 配置 | 历史准确率 | 本次准确率 | 差值（百分点） | 决策改变条数 |",
             "|---|---:|---:|---:|---:|"]
    for config in evaluation.prompts.CONFIGS:
        block = doc["suites"]["baseline18"][config]
        old_rows = [r for r in historical["cases"] if r["config"] == config]
        old_map = {(r["case_id"], r.get("dimension")): r for r in old_rows}
        changes = sum(r.get("pred_label", r.get("pred_bool")) != old_map[(r["case_id"], r.get("dimension"))].get("pred_label", old_map[(r["case_id"], r.get("dimension"))].get("pred_bool")) for r in block["cases"])
        old = historical["report"][config]["accuracy"]
        new = block["report"]["accuracy"]
        lines.append(f"| {config} | {old:.4f} | {new:.4f} | {(new-old)*100:+.2f} | {changes} |")
    lines += ["", "历史运行使用不同设备/依赖版本；这里只确认上述同权重同输入的决策差异，不把数值漂移归因于单一变量。", "",
              "## 新增 96 条：完整结果", "",
              "choice 的随机/多数类基线都是 1/6；noul 每条包含四个独立问题，按 384 个决策统计，不能与 choice 准确率直接比较。", "",
              "noul 全真基线为 33.33%，全假基线为 66.67%，逐维度多数类常数基线为 83.33%（三种动作均答假、is_command 答真）。", "",
              "| 配置 | 决策数 | 准确率 | Macro-F1 | ECE | 平均最大概率 | 重复决策改变 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for config, block in doc["suites"]["expanded96"].items():
        r = block["report"]
        f1 = f"{r['macro_f1']:.4f}" if 'macro_f1' in r else "—"
        ece = f"{r['ece']:.4f}" if 'ece' in r else "—"
        lines.append(f"| {config} | {len(block['cases'])} | {r['accuracy']:.4f} | {f1} | {ece} | {r['mean_confidence']:.4f} | {block['repeat_decision_changes']} |")
    lines += ["", "同任务形状内的变化（基础 choice 或 plain noul 为对照）：", "",
              "| 配置 | 准确率差（百分点） | 相对变化 |", "|---|---:|---:|"]
    for config, block in doc["suites"]["expanded96"].items():
        reference = "choice_criteria" if config in evaluation.prompts.CHOICE_CONFIGS else "noul_plain"
        before = doc["suites"]["expanded96"][reference]["report"]["accuracy"]
        after = block["report"]["accuracy"]
        lines.append(f"| {config} | {(after-before)*100:+.2f} | {(after/before-1)*100:+.2f}% |")
    lines += ["", "## choice 各标签和各现象", "",
              "各现象的类别组成不同，不能将组间差异直接解释为现象本身的因果效应。", "",
              "| 分组 | 数量 | criteria | scenario | JSON state |", "|---|---:|---:|---:|---:|"]
    cases = evaluation.datasets()["expanded96"]
    for field in ("gold", "family"):
        for value in sorted({c[field] for c in cases}):
            ids = {c["id"] for c in cases if c[field] == value}
            acc = [sum(r["correct"] for r in doc["suites"]["expanded96"][config]["cases"] if r["case_id"] in ids) / len(ids) for config in evaluation.prompts.CHOICE_CONFIGS]
            lines.append(f"| {field}:{value} | {len(ids)} | " + " | ".join(f"{a:.4f}" for a in acc) + " |")
    lines += ["", "## noul 各维度与正例比例", "",
              "每个维度 96 条。报告全真、全假基线，以避免把常数输出误认成能力。", "",
              "| 配置 | 维度 | 准确率 | 真值正例 | 预测正例 | 多数类基线 |", "|---|---|---:|---:|---:|---:|"]
    for config in evaluation.prompts.NOUL_CONFIGS:
        for dim in evaluation.prompts.NOUL_DIMENSIONS:
            rows = [r for r in doc["suites"]["expanded96"][config]["cases"] if r["dimension"] == dim]
            gold_true = sum(r["gold_bool"] for r in rows)
            pred_true = sum(r["pred_bool"] for r in rows)
            lines.append(f"| {config} | {dim} | {sum(r['correct'] for r in rows)/96:.4f} | {gold_true}/96 | {pred_true}/96 | {max(gold_true,96-gold_true)/96:.4f} |")
    lines += ["", "## 相反动作的配对诊断", "",
              "预先定义配对：ext-001..016 与 ext-017..032；ext-049..064 与 ext-065..080，共 32 对。", "",
              "| 配置 | 两条都正确 | 两条输出同一标签 |", "|---|---:|---:|"]
    for config in evaluation.prompts.CHOICE_CONFIGS:
        rows = {r["case_id"]: r for r in doc["suites"]["expanded96"][config]["cases"]}
        pairs = [(rows[f"ext-{a:03d}"], rows[f"ext-{a+16:03d}"]) for a in list(range(1,17))+list(range(49,65))]
        lines.append(f"| {config} | {sum(a['correct'] and b['correct'] for a,b in pairs)}/32 | {sum(a['pred_label']==b['pred_label'] for a,b in pairs)}/32 |")
    lines += ["", "## 基础配置错误样例", "", "| 输入 | 期望 | 实际 | 最大概率 |", "|---|---|---|---:|"]
    failures = [r for r in doc["suites"]["expanded96"]["choice_criteria"]["cases"] if not r["correct"]]
    for r in failures[:20]:
        lines.append(f"| {r['text']} | {r['gold_label']} | {r['pred_label']} | {r['confidence']:.4f} |")
    lines += ["", "## 最小 SDK 复现", "",
              "minimal_repro.py 直接调用公开 API，完整结果保存在 minimal_output.json。",
              "choice_criteria：太快了 → faster；太慢了 → slower；不要右转，请左转 → right。",
              "前两条来自原 18 条用例，第三条来自新增用例；它们是具体反例，不代表全部中文表达。", "",
              "## 结论与后续", "",
              "1. 观察：表格记录的是同一权重、冻结标签和原提示词下的行为；重复推理检查稳定性，不扩大样本量。",
              "2. 解释：否定、纠正动作和 noul 概率失去区分度是值得进一步检验的假设；本实验没有隔离模型、标签文字或提示词的因果贡献。",
              "3. 含义：用例可作为维护者复现问题的材料；不应直接替代业务数据验收或宣称整体中文能力。",
              "4. 下一步：补充独立多人标注并采集真实命令；在另一套预先冻结的数据上验证，分别控制选项顺序、标签文字和指令语言。", ""]
    total = sum(len(b['cases']) for s in doc['suites'].values() for b in s.values())
    lines += ["", "## 验证和交付", "",
              f"- {total} 条主运行决策及相同数量重复决策；14 个套件/配置组合",
              f"- SDK 对照 {len(doc['sdk_parity'])} 个答案，最大概率差 {max(r['delta'] for r in doc['sdk_parity']):.6f}",
              "- 审计从逐条 logits 重建请求、决策及指标；离线篡改测试验证拒绝错误归档",
              "- expanded_report.json 保留全部证据；predictions.csv 便于逐条查看",
              "- Issue 和 PR 材料为本地草稿，尚未发布或合并", "",
              "## 复现", "", "在项目根目录 PowerShell 执行：", "", "```powershell",
              '$env:PYTHONUTF8="1"', '$env:PYTHONDONTWRITEBYTECODE="1"', '$env:USE_TF="0"',
              '.\.venv\Scripts\python.exe -m unittest discover -s research/benchmarks/zh_short_commands/tests -v',
              '.\.venv\Scripts\python.exe research/benchmarks/zh_short_commands/expanded_eval.py --device cpu --out research/benchmarks/zh_short_commands/expanded_rerun.json',
              '.\.venv\Scripts\python.exe research/benchmarks/zh_short_commands/expanded_eval.py --audit research/benchmarks/zh_short_commands/expanded_report.json', "```", "",
              "环境 .venv 复用 D:/Anaconda/envs/langchain310 的 PyTorch；无需 Codex 运行时，保留基础环境即可运行。"]
    (here / "REPORT.zh-CN.md").write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")
    records = [dict(suite=suite, **r) for suite, blocks in doc["suites"].items() for block in blocks.values() for r in block["cases"]]
    fields = ["suite", "config", "case_id", "text", "family", "dimension", "gold_label", "gold_bool", "pred_label", "pred_bool", "p_true", "confidence", "correct"]
    with open(here / "predictions.csv", "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    config_lines = [f"| {c} | {b['report']['accuracy']:.4f} |" for c,b in doc["suites"]["expanded96"].items()]
    examples = "\n".join(f"- `{r['text']}`: expected `{r['gold_label']}`, got `{r['pred_label']}` (max probability {r['confidence']:.4f})." for r in failures[:5])
    issue = f"""# Chinese movement-command diagnostic: frozen 96-case extension

This extends the existing 18-case zh_short_commands diagnostic with 96 frozen assistant-authored synthetic fixtures, balanced across six labels. It is a diagnostic contribution, not a population-level Chinese benchmark. Labels were assigned before inference; there was no fitting or prompt selection. Existing prompts and archives are preserved.

Related prior report: https://github.com/NandhaKishorM/laya/issues/218 (closed). This extends the archived investigation rather than reporting the same issue as undiscovered.

## Environment
Laya {metadata['laya']}; commit `{metadata['git_commit']}`; Python {metadata['python'].split()[0]}; torch {metadata['torch']}; transformers {metadata['transformers']}; Windows CPU FP32; deterministic algorithms; four intra-op threads and one inter-op thread.
Weight SHA-256: `{metadata['checkpoint']['weight_sha256']}`.

## Results on the additional fixtures
| Config | Accuracy |
|---|---:|
{chr(10).join(config_lines)}

Choice has 96 decisions; noul has 384 decisions per configuration. These accuracies should not be compared across task shapes. Noul constant baselines: all-true 0.3333, all-false 0.6667, per-dimension majority 0.8333. Each run was repeated twice, with zero decision changes. Selected SDK answers were checked against the research scorer. The original 18 cases were run separately; their per-config historical comparison is in the attached report.

## Concrete examples (choice_criteria)
{examples}

## Reproduce
Use the frozen cases, manifest, runner, JSON archive and offline tests in this contribution.
```powershell
$env:PYTHONUTF8="1"
$env:USE_TF="0"
.\\.venv\\Scripts\\python.exe research/benchmarks/zh_short_commands/expanded_eval.py --device cpu --out research/benchmarks/zh_short_commands/expanded_rerun.json
.\\.venv\\Scripts\\python.exe research/benchmarks/zh_short_commands/expanded_eval.py --audit research/benchmarks/zh_short_commands/expanded_report.json
```

Could this focused diagnostic extension be useful upstream? It records raw logits and repeat runs, checks integrity offline, and does not change inference behavior. Similar paraphrases are correlated; all 96 fixtures and labels were reviewed and approved unchanged by the contributor on 2026-10-13, after evaluation and before publication. This is a single-contributor review, not independent multi-annotator validation. We have not established whether any observed error originates in the checkpoint, prompt wording, or another component.
"""
    (here / "ISSUE_DRAFT.md").write_text(issue, encoding="utf-8", newline="\n")
    print("Wrote Chinese report, per-decision CSV, and English issue draft")


if __name__ == "__main__":
    main()
