import { RISK_LEVELS } from "../../../utils/labels";

export default function FilterPanel({
  departments,
  tenureRange,
  filters,
  setFilters,
  onReset,
}) {
  const [minTenure, maxTenure] = tenureRange;

  return (
    <section className="panel">
      <h2>Filters</h2>

      <label className="field">
        <span>Department</span>
        <select
          value={filters.department}
          onChange={(event) => setFilters((prev) => ({ ...prev, department: event.target.value }))}
        >
          <option value="All">All departments</option>
          {departments.map((dept) => (
            <option key={dept} value={dept}>
              {dept}
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        <span>Risk Level</span>
        <select
          value={filters.riskLevel}
          onChange={(event) => setFilters((prev) => ({ ...prev, riskLevel: event.target.value }))}
        >
          <option value="All">All levels</option>
          {RISK_LEVELS.map((risk) => (
            <option key={risk} value={risk}>
              {risk}
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        <span>Min Tenure ({filters.minTenure.toFixed(1)})</span>
        <input
          type="range"
          min={minTenure}
          max={maxTenure}
          step="0.5"
          value={filters.minTenure}
          onChange={(event) => setFilters((prev) => ({ ...prev, minTenure: Number(event.target.value) }))}
        />
      </label>

      <button type="button" className="ghost-btn" onClick={onReset}>
        Reset Filters
      </button>
    </section>
  );
}
