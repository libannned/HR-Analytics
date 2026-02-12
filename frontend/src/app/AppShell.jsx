export default function AppShell({
  title,
  subtitle,
  tabs,
  activeTab,
  onTabChange,
  loading,
  error,
  rowCount,
  dataQuality,
  children,
}) {
  return (
    <div className="page">
      <header className="topbar">
        <div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        <div className="topbar-meta">
          <div className="meta-pill">Rows: {rowCount.toLocaleString()}</div>
          <div className="meta-pill">Missing Values: {dataQuality.missingPercent}%</div>
        </div>
      </header>

      <nav className="tabs" aria-label="Dashboard Tabs">
        {tabs.map((tab) => (
          <button
            key={tab}
            type="button"
            className={`tab ${activeTab === tab ? "active" : ""}`}
            onClick={() => onTabChange(tab)}
          >
            {tab}
          </button>
        ))}
      </nav>

      <main className="content">
        {loading ? <div className="status-card">Loading dataset...</div> : null}
        {error ? <div className="status-card error">Error: {error}</div> : null}
        {!loading && !error ? children : null}
      </main>
    </div>
  );
}
