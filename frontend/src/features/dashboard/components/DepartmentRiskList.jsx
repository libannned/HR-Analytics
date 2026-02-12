export default function DepartmentRiskList({ data }) {
  const grouped = new Map();

  data.forEach((row) => {
    const key = row.department || "Unknown";
    const prev = grouped.get(key) || { total: 0, riskSum: 0, highRisk: 0 };
    prev.total += 1;
    prev.riskSum += row.riskScore;
    if (row.riskLevel === "High") prev.highRisk += 1;
    grouped.set(key, prev);
  });

  const rows = Array.from(grouped.entries())
    .map(([department, stats]) => ({
      department,
      avgRisk: stats.riskSum / stats.total,
      highRisk: stats.highRisk,
      total: stats.total,
    }))
    .sort((a, b) => b.avgRisk - a.avgRisk)
    .slice(0, 8);

  return (
    <section className="panel">
      <h2>Departments by Average Risk</h2>
      <ul className="list">
        {rows.map((row) => (
          <li key={row.department} className="list-row">
            <div>
              <strong>{row.department}</strong>
              <p>{row.highRisk} high-risk out of {row.total}</p>
            </div>
            <span className="pill">{row.avgRisk.toFixed(2)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
