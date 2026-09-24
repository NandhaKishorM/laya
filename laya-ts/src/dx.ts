import type { QuestionDef, SystemAnswer } from "./agent.js";

/** One-line question factories; output always passes checkQuestion. */
export const q = {
  choice(instructions: string, criteria: Record<string, unknown> | string[]): QuestionDef {
    return { type: "choice", instructions, criteria } as QuestionDef;
  },
  score(instructions: string, criteria: unknown[]): QuestionDef {
    return { type: "score", instructions, criteria } as QuestionDef;
  },
  noul(instructions: string, criteria?: Record<string, unknown>, labels?: { false: string; true: string }): QuestionDef {
    const out: Record<string, unknown> = { type: "noul", instructions };
    if (criteria !== undefined) out["criteria"] = criteria;
    if (labels !== undefined) out["labels"] = labels;
    return out as QuestionDef;
  },
};

/** Argmax label: choice string, score key, noul boolean. */
export function best(answer: SystemAnswer): string | boolean {
  if (answer.type === "choice") return answer.choice;
  if (answer.type === "score") {
    let bk = "";
    let bv = -Infinity;
    for (const [k, v] of Object.entries(answer.probabilities)) {
      if (v > bv) {
        bv = v;
        bk = k;
      }
    }
    return bk;
  }
  return answer.noul >= 0.5;
}

/** Top-k probability keys, descending. */
export function topK(probs: Record<string, number>, k: number): string[] {
  return Object.entries(probs)
    .sort((a, b) => b[1] - a[1])
    .slice(0, Math.max(0, k))
    .map(([key]) => key);
}

/** Confidence gate (default 0.8). Reads answer_confidence (max(p), comparable
 * across question types) when present, falling back to entropy confidence. */
export function isConfident(
  answer: Pick<SystemAnswer, "confidence"> & { answer_confidence?: unknown },
  thresh = 0.8,
): boolean {
  const ac = (answer as { answer_confidence?: unknown }).answer_confidence;
  const v = typeof ac === "number" && Number.isFinite(ac) ? ac : answer.confidence;
  return (v ?? 0) >= thresh;
}
