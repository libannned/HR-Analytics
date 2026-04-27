export default function DemoChecklist() {
  const steps = [
    "Show KPI cards: employees in view, high-risk count, average risk score.",
    "Apply department + risk filters and explain how charts update.",
    "Open At-Risk tab and review suggested focus actions.",
    "Open Scenario tab, adjust inputs, and explain risk level changes.",
  ];

  return (
    <section className="panel demo-panel">
      <h2>Live Demo Checklist</h2>
      <ol className="bullet-list ordered">
        {steps.map((step) => (
          <li key={step}>{step}</li>
        ))}
      </ol>
    </section>
  );
}
