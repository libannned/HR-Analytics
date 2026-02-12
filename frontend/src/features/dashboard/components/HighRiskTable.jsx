function recommendationSummary(row) {
  const notes = [];
  if ((row.anxiety ?? 0) >= 7) notes.push("Stress support");
  if ((row.selfControl ?? 10) <= 4) notes.push("Clearer goals");
  if ((row.tenure ?? 999) < 12) notes.push("Onboarding support");
  if (row.coachingSupport === "no") notes.push("Assign a coach");
  if (row.compensationType === "grey") notes.push("Compensation review");
  return notes.length ? notes.join(" | ") : "Stay interview";
}

export default function HighRiskTable({ data }) {
  const rows = data
    .filter((row) => row.riskLevel === "High")
    .sort((a, b) => b.riskScore - a.riskScore)
    .slice(0, 25);

  return (
    <section className="panel">
      <h2>High-Risk Employees</h2>
      {rows.length === 0 ? (
        <p>No high-risk employees under current filters.</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Department</th>
                <th>Role</th>
                <th>Tenure</th>
                <th>Risk</th>
                <th>Suggested Focus</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id}>
                  <td>{row.id}</td>
                  <td>{row.department}</td>
                  <td>{row.jobFunction}</td>
                  <td>{row.tenure?.toFixed(1) ?? "-"}</td>
                  <td>
                    <span className="risk-badge high">{(row.riskScore * 100).toFixed(0)}%</span>
                  </td>
                  <td>{recommendationSummary(row)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
