/**
 * ARCPanel
 * ========
 * Displays current Adaptive Retrieval Controller parameters as animated bars.
 * Polls /arc-status every 4 seconds and updates live.
 */

import { useState, useEffect } from "react";

import { API } from "./config";

function ARCPanel({ externalParams }) {
  const [params, setParams] = useState({
    similarity_threshold: 0.45,
    top_k: 3,
    chunk_size: 1000,
    chunk_overlap: 200,
    adjustment_count: 0,
  });

  // Poll backend for live ARC state
  useEffect(() => {
    const fetchParams = async () => {
      try {
        const res = await fetch(`${API}/arc-status`);
        if (res.ok) {
          const data = await res.json();
          setParams(data);
        }
      } catch (_) {
        // backend not running yet — keep defaults
      }
    };

    fetchParams();
    const id = setInterval(fetchParams, 4000);
    return () => clearInterval(id);
  }, []);

  // Allow parent to push params from SSE events (takes priority)
  useEffect(() => {
    if (externalParams && Object.keys(externalParams).length > 0) {
      setParams((prev) => ({ ...prev, ...externalParams }));
    }
  }, [externalParams]);

  const bars = [
    {
      label: "Similarity Threshold",
      value: params.similarity_threshold,
      display: params.similarity_threshold?.toFixed(3),
      min: 0.35,
      max: 0.60,
      pct:
        ((params.similarity_threshold - 0.35) / (0.60 - 0.35)) * 100,
    },
    {
      label: "Top-K Retrieval",
      value: params.top_k,
      display: params.top_k,
      min: 2,
      max: 8,
      pct: ((params.top_k - 2) / (8 - 2)) * 100,
    },
    {
      label: "Chunk Size",
      value: params.chunk_size,
      display: `${params.chunk_size} tok`,
      min: 300,
      max: 2000,
      pct: ((params.chunk_size - 300) / (2000 - 300)) * 100,
    },
    {
      label: "Chunk Overlap",
      value: params.chunk_overlap,
      display: `${params.chunk_overlap} tok`,
      min: 50,
      max: 400,
      pct: ((params.chunk_overlap - 50) / (400 - 50)) * 100,
    },
  ];

  return (
    <div className="arc-panel">
      {bars.map((bar) => (
        <div key={bar.label} className="arc-param">
          <div className="arc-param-header">
            <span className="arc-param-label">{bar.label}</span>
            <span className="arc-param-value">{bar.display}</span>
          </div>
          <div className="arc-bar-track">
            <div
              className="arc-bar-fill"
              style={{ width: `${Math.max(4, bar.pct)}%` }}
            />
          </div>
        </div>
      ))}

      <div className="arc-adj-count">
        <span>Dynamic adjustments:</span>
        <span className="arc-adj-badge">{params.adjustment_count ?? 0}</span>
      </div>
    </div>
  );
}

export default ARCPanel;
