export default function GuidePanel() {
  return (
    <section className="guide-grid">
      <article className="panel">
        <h2>How to Read Risk</h2>
        <ul className="bullet-list">
          <li><strong>Low (&lt; 0.40):</strong> normal monitoring.</li>
          <li><strong>Medium (0.40-0.69):</strong> early intervention and manager check-ins.</li>
          <li><strong>High (&gt;= 0.70):</strong> immediate retention planning.</li>
        </ul>
      </article>

      <article className="panel">
        <h2>Plain-English Factors</h2>
        <ul className="bullet-list">
          <li><strong>Independence:</strong> self-directed work style.</li>
          <li><strong>Self-Control:</strong> consistency and discipline in execution.</li>
          <li><strong>Anxiety:</strong> tendency to experience stress.</li>
          <li><strong>Innovation Openness:</strong> comfort with change/new ideas.</li>
          <li><strong>Coaching Support:</strong> whether guidance exists.</li>
        </ul>
      </article>

      <article className="panel">
        <h2>Recommended Usage</h2>
        <ol className="bullet-list ordered">
          <li>Filter for high-risk employees in one department.</li>
          <li>Review top contributing context factors.</li>
          <li>Apply one concrete retention action and track outcomes.</li>
        </ol>
      </article>
    </section>
  );
}
