import { parseCsv } from "../utils/csv";
import { calculateHeuristicRisk, scoreToRiskLevel } from "../utils/risk";
import { prettyValue } from "../utils/labels";

function numberOrNull(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

export function normalizeRow(raw, index) {
  const record = {
    id: index + 1,
    tenure: numberOrNull(raw.stag),
    turnoverEvent: Number(raw.event) === 1 ? 1 : 0,
    gender: raw.gender || "",
    age: numberOrNull(raw.age),
    department: raw.industry || "Unknown",
    jobFunction: raw.profession || "Unknown",
    recruitmentSource: raw.traffic || "",
    coachingSupport: raw.coach || "",
    managerGender: raw.head_gender || "",
    compensationType: raw.greywage || "",
    commuteMethod: raw.way || "",
    extraversion: numberOrNull(raw.extraversion),
    independence: numberOrNull(raw.independ),
    selfControl: numberOrNull(raw.selfcontrol),
    anxiety: numberOrNull(raw.anxiety),
    innovationOpenness: numberOrNull(raw.novator),
  };

  const riskScore = calculateHeuristicRisk(record);
  record.riskScore = riskScore;
  record.riskLevel = scoreToRiskLevel(riskScore);
  record.recruitmentSourceLabel = prettyValue("recruitmentSource", record.recruitmentSource);
  record.coachingSupportLabel = prettyValue("coachingSupport", record.coachingSupport);
  record.compensationTypeLabel = prettyValue("compensationType", record.compensationType);
  return record;
}

export async function fetchTurnoverData() {
  const response = await fetch("/data/turnover.csv");
  if (!response.ok) {
    throw new Error(`Failed to load data file: ${response.status}`);
  }

  const text = await response.text();
  const rawRows = parseCsv(text);
  return rawRows.map((row, index) => normalizeRow(row, index));
}

export function calculateDataQuality(data) {
  if (!data.length) {
    return { missingPercent: 0 };
  }

  const fields = Object.keys(data[0]);
  let missing = 0;
  let total = 0;

  data.forEach((row) => {
    fields.forEach((field) => {
      total += 1;
      if (row[field] === null || row[field] === undefined || row[field] === "") {
        missing += 1;
      }
    });
  });

  return {
    missingPercent: ((missing / total) * 100).toFixed(1),
  };
}
