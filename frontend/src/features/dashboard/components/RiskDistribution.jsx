const LEVELS = ["Low", "Medium", "High"];

export default function RiskDistribution({ data }) {
  const counts = LEVELS.map((level) => ({
    level,
    count: data.filter((row) => row.riskLevel === level).length,
  }));

  const max = Math.max(...counts.map((item) => item.count), 1);

  return (
    <section className="panel">
      <h2>Risk Distribution</h2>
      <div className="bars">
        {counts.map((item) => (
          <div key={item.level} className="bar-row">
            <div className="bar-label">{item.level}</div>
            <div className="bar-track">
              <div
                className={`bar-fill ${item.level.toLowerCase()}`}
                style={{ width: `${(item.count / max) * 100}%` }}
              />
            </div>
            <div className="bar-value">{item.count}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
