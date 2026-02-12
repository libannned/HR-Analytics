import { useMemo, useState } from "react";
import { calculateHeuristicRisk, scoreToRiskLevel } from "../../utils/risk";
import { prettyValue } from "../../utils/labels";

function riskMessage(level) {
  if (level === "High") return "Immediate support plan recommended.";
  if (level === "Medium") return "Early intervention recommended.";
  return "Maintain normal engagement and check-ins.";
}

export default function PredictorPanel({ sourceData }) {
  const defaults = useMemo(() => {
    if (!sourceData.length) {
      return {
        tenure: 12,
        anxiety: 5,
        selfControl: 5,
        coachingSupport: "yes",
        compensationType: "white",
      };
    }

    const avg = (field, fallback = 5) => {
      const values = sourceData.map((x) => x[field]).filter((x) => Number.isFinite(x));
      if (!values.length) return fallback;
      return values.reduce((a, b) => a + b, 0) / values.length;
    };

    return {
      tenure: avg("tenure", 12),
      anxiety: avg("anxiety", 5),
      selfControl: avg("selfControl", 5),
      coachingSupport: "yes",
      compensationType: "white",
    };
  }, [sourceData]);

  const [form, setForm] = useState(defaults);

  const score = calculateHeuristicRisk(form);
  const level = scoreToRiskLevel(score);

  return (
    <section className="panel">
      <h2>Scenario Predictor</h2>
      <p className="muted">Adjust factors to explore how risk could change for a hypothetical employee.</p>

      <div className="form-grid">
        <label className="field">
          <span>Tenure (months)</span>
          <input
            type="number"
            min="0"
            value={form.tenure}
            onChange={(e) => setForm((p) => ({ ...p, tenure: Number(e.target.value) }))}
          />
        </label>

        <label className="field">
          <span>Anxiety (0-10)</span>
          <input
            type="range"
            min="0"
            max="10"
            step="0.1"
            value={form.anxiety}
            onChange={(e) => setForm((p) => ({ ...p, anxiety: Number(e.target.value) }))}
          />
        </label>

        <label className="field">
          <span>Self-Control (0-10)</span>
          <input
            type="range"
            min="0"
            max="10"
            step="0.1"
            value={form.selfControl}
            onChange={(e) => setForm((p) => ({ ...p, selfControl: Number(e.target.value) }))}
          />
        </label>

        <label className="field">
          <span>Coaching Support</span>
          <select
            value={form.coachingSupport}
            onChange={(e) => setForm((p) => ({ ...p, coachingSupport: e.target.value }))}
          >
            <option value="yes">{prettyValue("coachingSupport", "yes")}</option>
            <option value="my head">{prettyValue("coachingSupport", "my head")}</option>
            <option value="no">{prettyValue("coachingSupport", "no")}</option>
          </select>
        </label>

        <label className="field">
          <span>Compensation Type</span>
          <select
            value={form.compensationType}
            onChange={(e) => setForm((p) => ({ ...p, compensationType: e.target.value }))}
          >
            <option value="white">{prettyValue("compensationType", "white")}</option>
            <option value="grey">{prettyValue("compensationType", "grey")}</option>
          </select>
        </label>
      </div>

      <div className="result-card">
        <p>Predicted Risk Score</p>
        <h3>{(score * 100).toFixed(1)}%</h3>
        <span className={`risk-badge ${level.toLowerCase()}`}>{level} Risk</span>
        <p className="muted">{riskMessage(level)}</p>
      </div>
    </section>
  );
}
