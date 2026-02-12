const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

export function scoreToRiskLevel(score) {
  if (score < 0.4) return "Low";
  if (score < 0.7) return "Medium";
  return "High";
}

export function calculateHeuristicRisk(record) {
  const anxiety = Number(record.anxiety || 0) / 10;
  const selfControl = Number(record.selfControl || 0) / 10;
  const tenure = Number(record.tenure || 0);

  const tenurePenalty = tenure < 12 ? 0.22 : tenure < 24 ? 0.12 : 0.03;
  const coachPenalty = record.coachingSupport === "no" ? 0.11 : 0;
  const wagePenalty = record.compensationType === "grey" ? 0.08 : 0;

  const score =
    0.32 * anxiety +
    0.28 * (1 - selfControl) +
    tenurePenalty +
    coachPenalty +
    wagePenalty;

  return clamp(score, 0.02, 0.98);
}
