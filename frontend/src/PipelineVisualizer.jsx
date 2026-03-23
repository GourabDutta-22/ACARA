/**
 * PipelineVisualizer
 * ==================
 * Live flow diagram matching the Adaptive Context-Aware Retrieval Architecture.
 * Nodes glow/pulse when active, turn green when done.
 *
 * Props:
 *   activeStep  — string key of the currently executing node
 *   doneSteps   — array of completed step keys
 *   webFallback — boolean: did the pipeline take the web search path?
 */

const NODES = [
  {
    key: "retrieve",
    icon: "🔍",
    name: "Query Encoder",
    desc: "Vector Memory lookup",
  },
  {
    key: "grade",
    icon: "⚖️",
    name: "Context Awareness Gate",
    desc: "Similarity · Coverage · Freshness",
  },
  // ── Context Weak branch ──
  {
    key: "web_search",
    icon: "🌐",
    name: "External Knowledge Source",
    desc: "Tavily web search",
    branch: "weak",
  },
  {
    key: "chunk",
    icon: "✂️",
    name: "Dynamic Chunking Module",
    desc: "Adaptive size · Semantic split · Overlap",
    branch: "weak",
  },
  {
    key: "embed",
    icon: "🔢",
    name: "Embedding Model",
    desc: "Vectorising chunks",
    branch: "weak",
  },
  {
    key: "credibility",
    icon: "🛡️",
    name: "Credibility Scoring",
    desc: "Filters low-quality chunks",
    branch: "weak",
  },
  {
    key: "memory_update",
    icon: "💾",
    name: "Memory Update",
    desc: "Writing to Vector Store",
    branch: "weak",
  },
  // ── Context Strong branch ──
  {
    key: "context_builder",
    icon: "🏗️",
    name: "Context Builder",
    desc: "Assembles ranked context",
    branch: "strong",
  },
  // ── Shared final nodes ──
  {
    key: "generate",
    icon: "🤖",
    name: "Generator Model (LLM)",
    desc: "GPT-4o-mini synthesis",
  },
  {
    key: "critic",
    icon: "🔎",
    name: "Critic / Validator",
    desc: "Hallucination check",
  },
  {
    key: "done",
    icon: "✅",
    name: "Final Output",
    desc: "Answer delivered",
  },
];

// ARC is shown as a separate callout
const ARC_NODE = {
  key: "arc",
  icon: "⚙️",
  name: "Adaptive Retrieval Controller",
  desc: "Adjusts threshold · top-k · chunk size",
};

function PipelineVisualizer({ activeStep, doneSteps = [], webFallback }) {
  const getState = (key) => {
    if (activeStep === key) return "active";
    if (doneSteps.includes(key)) return "done";

    // Dim nodes that belong to the other branch
    const isWeakBranch = NODES.find((n) => n.key === key)?.branch === "weak";
    const isStrongBranch =
      NODES.find((n) => n.key === key)?.branch === "strong";

    if (webFallback !== undefined) {
      if (webFallback && isStrongBranch) return "skipped";
      if (!webFallback && isWeakBranch) return "skipped";
    }

    return "idle";
  };

  return (
    <div className="pipeline-visualizer">
      {/* ARC callout at top */}
      <div
        className={`pipeline-node ${
          doneSteps.length > 0 ? "done" : "idle"
        }`}
        style={{ marginBottom: 4 }}
      >
        <span className="node-icon">{ARC_NODE.icon}</span>
        <div className="node-info">
          <div className="node-name" style={{ fontSize: 11 }}>
            {ARC_NODE.name}
          </div>
          <div className="node-desc">{ARC_NODE.desc}</div>
        </div>
        <span className="node-status-dot" />
      </div>

      <div className="pipeline-connector" />

      {NODES.map((node, i) => {
        const state = getState(node.key);
        const prevNode = NODES[i - 1];

        return (
          <div key={node.key}>
            {/* Branch label */}
            {node.branch === "weak" && prevNode?.branch !== "weak" && (
              <div className="pipeline-connector branch">
                <span style={{ color: "var(--accent-danger)", fontWeight: 600 }}>
                  ↙ Context Weak
                </span>
              </div>
            )}
            {node.branch === "strong" && prevNode?.branch !== "strong" && (
              <div className="pipeline-connector branch">
                <span
                  style={{
                    color: "var(--accent-success)",
                    fontWeight: 600,
                  }}
                >
                  ↘ Context Strong
                </span>
              </div>
            )}
            {/* Connector line */}
            {i > 0 &&
              node.branch === prevNode?.branch &&
              !node.branch && (
                <div className="pipeline-connector" />
              )}
            {i > 0 && node.branch && node.branch === prevNode?.branch && (
              <div className="pipeline-connector" />
            )}

            <div className={`pipeline-node ${state}`}>
              <span className="node-icon">{node.icon}</span>
              <div className="node-info">
                <div className="node-name">{node.name}</div>
                <div className="node-desc">{node.desc}</div>
              </div>
              <span className="node-status-dot" />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default PipelineVisualizer;
