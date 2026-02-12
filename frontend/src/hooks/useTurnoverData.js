import { useEffect, useMemo, useState } from "react";
import { calculateDataQuality, fetchTurnoverData } from "../services/dataService";

export function useTurnoverData() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;

    async function load() {
      setLoading(true);
      setError("");
      try {
        const rows = await fetchTurnoverData();
        if (alive) setData(rows);
      } catch (err) {
        if (alive) setError(err.message || "Unknown error while loading dataset.");
      } finally {
        if (alive) setLoading(false);
      }
    }

    load();
    return () => {
      alive = false;
    };
  }, []);

  const dataQuality = useMemo(() => calculateDataQuality(data), [data]);

  return {
    data,
    loading,
    error,
    dataQuality,
  };
}
