import { useMemo, useState } from "react";
import AppShell from "./AppShell";
import DashboardPage from "../features/dashboard/DashboardPage";
import PredictorPanel from "../features/predictor/PredictorPanel";
import GuidePanel from "../features/guide/GuidePanel";
import { useTurnoverData } from "../hooks/useTurnoverData";

const TABS = ["Overview", "At-Risk", "Scenario", "Guide"];

export default function App() {
  const { data, loading, error, dataQuality } = useTurnoverData();
  const [activeTab, setActiveTab] = useState("Overview");

  const defaultFilters = useMemo(() => {
    if (!data.length) {
      return { departments: [], riskLevels: ["Low", "Medium", "High"], tenureRange: [0, 100] };
    }

    const tenures = data.map((row) => row.tenure).filter((v) => Number.isFinite(v));
    return {
      departments: Array.from(new Set(data.map((row) => row.department).filter(Boolean))).sort(),
      riskLevels: ["Low", "Medium", "High"],
      tenureRange: [Math.min(...tenures), Math.max(...tenures)],
    };
  }, [data]);

  return (
    <AppShell
      title="HR Turnover Risk Explorer"
      subtitle="A personal project prototype for exploring employee turnover signals"
      tabs={TABS}
      activeTab={activeTab}
      onTabChange={setActiveTab}
      loading={loading}
      error={error}
      rowCount={data.length}
      dataQuality={dataQuality}
    >
      {activeTab === "Guide" ? (
        <GuidePanel />
      ) : activeTab === "Scenario" ? (
        <PredictorPanel sourceData={data} />
      ) : (
        <DashboardPage sourceData={data} activeTab={activeTab} defaultFilters={defaultFilters} />
      )}
    </AppShell>
  );
}
