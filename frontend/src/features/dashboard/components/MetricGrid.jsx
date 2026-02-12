export default function MetricGrid({ items }) {
  return (
    <section className="metric-grid">
      {items.map((item) => (
        <article key={item.label} className="metric-card">
          <p className="metric-label">{item.label}</p>
          <h3 className="metric-value">{item.value}</h3>
          {item.note ? <p className="metric-note">{item.note}</p> : null}
        </article>
      ))}
    </section>
  );
}
