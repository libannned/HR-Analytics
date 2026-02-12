import { useMemo, useState } from "react";
import FilterPanel from "./components/FilterPanel";
import MetricGrid from "./components/MetricGrid";
import RiskDistribution from "./components/RiskDistribution";
import DepartmentRiskList from "./components/DepartmentRiskList";
import HighRiskTable from "./components/HighRiskTable";

export default function DashboardPage({ sourceData, activeTab, defaultFilters }) {
  const [filters, setFilters] = useState({
    department: "All",
    riskLevel: "All",
    minTenure: defaultFilters.tenureRange[0] ?? 0,
  });

  const filteredData = useMemo(
    () =>
      sourceData.filter((row) => {
        if (filters.department !== "All" && row.department !== filters.department) return false;
        if (filters.riskLevel !== "All" && row.riskLevel !== filters.riskLevel) return false;
        if ((row.tenure ?? 0) < filters.minTenure) return false;
        return true;
      }),
    [sourceData, filters]
  );

  const metrics = useMemo(() => {
    const total = filteredData.length;
    const highRisk = filteredData.filter((row) => row.riskLevel === "High").length;
    const avgRisk =
      total > 0
        ? filteredData.reduce((sum, row) => sum + row.riskScore, 0) / total
        : 0;

    const turnoverRate =
      total > 0
        ? (filteredData.filter((row) => row.turnoverEvent === 1).length / total) * 100
        : 0;

    return [
      { label: "Employees in View", value: total.toLocaleString() },
      { label: "High-Risk Employees", value: highRisk.toLocaleString(), note: "Risk >= 0.70" },
      { label: "Average Risk Score", value: avgRisk.toFixed(2) },
      { label: "Observed Turnover Rate", value: `${turnoverRate.toFixed(1)}%` },
    ];
  }, [filteredData]);

  const tenureRange = defaultFilters.tenureRange;

  const resetFilters = () => {
    setFilters({
      department: "All",
      riskLevel: "All",
      minTenure: tenureRange[0] ?? 0,
    });
  };

  return (
    <section className="dashboard-layout">
      <aside>
        <FilterPanel
          departments={defaultFilters.departments}
          tenureRange={tenureRange}
          filters={filters}
          setFilters={setFilters}
          onReset={resetFilters}
        />
      </aside>
      <section className="dashboard-main">
        <MetricGrid items={metrics} />
        <div className="two-col">
          <RiskDistribution data={filteredData} />
          <DepartmentRiskList data={filteredData} />
        </div>
        {activeTab === "At-Risk" ? <HighRiskTable data={filteredData} /> : null}
      </section>
    </section>
  );
}
