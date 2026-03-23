/**
 * App.jsx — Adaptive Context-Aware RAG
 * ======================================
 * 3-panel layout:
 *   Left  — Sidebar (chat history + stats)
 *   Center — Chat interface with markdown + feedback
 *   Right  — Pipeline Visualizer + ARC Panel + Upload Panel
 *
 * Uses SSE (EventSource) from /stream-chat for real-time pipeline events.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import PipelineVisualizer from "./PipelineVisualizer";
import ARCPanel from "./ARCPanel";
import UploadPanel from "./UploadPanel";

import { API } from "./config";

const SUGGESTED_QUERIES = [
  "What is LangGraph and how does it work?",
  "Explain adaptive retrieval in RAG systems",
  "How does vector similarity search work?",
  "What are the benefits of dynamic chunking?",
];

let sessionCounter = 1;

function App() {
  // ── Chat state ─────────────────────────────────────────────────────────
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [waitingForFeedback, setWaitingForFeedback] = useState(false);
  const [currentThread, setCurrentThread] = useState(null);

  // ── Chat history (sidebar) ─────────────────────────────────────────────
  const [chatHistory, setChatHistory] = useState([]);
  const [activeChatId, setActiveChatId] = useState("default");
  const [historyLoading, setHistoryLoading] = useState(false);

  // ── Pipeline state ─────────────────────────────────────────────────────
  const [activeStep, setActiveStep] = useState(null);
  const [doneSteps, setDoneSteps] = useState([]);
  const [webFallback, setWebFallback] = useState(undefined);
  const [currentStepLabel, setCurrentStepLabel] = useState("");
  const [arcParams, setArcParams] = useState({});

  // ── Stats ──────────────────────────────────────────────────────────────
  const [vectorDocCount, setVectorDocCount] = useState("—");
  const [showUploadModal, setShowUploadModal] = useState(false);

  // ── UI toggles ─────────────────────────────────────────────────────────
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [showSplash, setShowSplash] = useState(true);

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // ── Get current session ID ─────────────────────────────────────────────
  const sessionId = chatHistory.find((c) => c.id === activeChatId)?.sessionId || "session-1";

  // ── Auto scroll & Splash Timer ─────────────────────────────────────────
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    const timer = setTimeout(() => setShowSplash(false), 3000);
    return () => clearTimeout(timer);
  }, []);

  // ── Fetch vector store stats ───────────────────────────────────────────
  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API}/stats`);
      if (res.ok) {
        const data = await res.json();
        setVectorDocCount(data.document_count);
      }
    } catch (_) {}
  }, []);

  useEffect(() => {
    fetchStats();
    const id = setInterval(fetchStats, 8000);
    return () => clearInterval(id);
  }, [fetchStats]);

  // ── Fetch sessions from backend ────────────────────────────────────────
  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch(`${API}/sessions`);
      if (res.ok) {
        const data = await res.json();
        if (data.sessions && data.sessions.length > 0) {
          const loaded = data.sessions.map((s) => ({
            id: s.session_id,       // use session_id as the item id
            title: s.title || "Untitled Chat",
            sessionId: s.session_id,
          }));
          setChatHistory(loaded);
          // Activate the most-recent session on first load
          setActiveChatId((prev) =>
            prev === "default" ? loaded[loaded.length - 1].id : prev
          );
        }
      }
    } catch (_) {}
  }, []);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  // ── New chat ───────────────────────────────────────────────────────────
  const handleNewChat = async () => {
    const id = String(Date.now());
    sessionCounter++;
    const newSession = `session-${sessionCounter}`;
    setChatHistory((prev) => [
      { id, title: "New Chat", sessionId: newSession },
      ...prev,
    ]);
    setActiveChatId(id);
    setMessages([]);
    setActiveStep(null);
    setDoneSteps([]);
    setWebFallback(undefined);
    setWaitingForFeedback(false);
    setCurrentThread(null);
    setArcParams({}); // clear UI params

    try {
      await fetch(`${API}/arc/reset`, { method: "POST" });
    } catch (_) {}
  };

  // ── Switch chat ────────────────────────────────────────────────────────
  const handleSwitchChat = async (id) => {
    setActiveChatId(id);
    setActiveStep(null);
    setDoneSteps([]);
    setWebFallback(undefined);
    setWaitingForFeedback(false);
    setCurrentThread(null);

    // Load messages for this session from backend
    const chat = chatHistory.find((c) => c.id === id);
    if (!chat) { setMessages([]); return; }
    setHistoryLoading(true);
    try {
      const res = await fetch(`${API}/history/${chat.sessionId}`);
      if (res.ok) {
        const data = await res.json();
        setMessages(data.messages || []);
      } else {
        setMessages([]);
      }
    } catch (_) {
      setMessages([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  // ── Delete chat ────────────────────────────────────────────────────────
  const handleDeleteChat = async (e, id) => {
    e.stopPropagation();
    
    // Optimistic UI update
    setChatHistory((prev) => prev.filter((c) => c.id !== id));
    
    // If the active chat was deleted, switch to a new chat
    if (activeChatId === id) {
      handleNewChat();
    }

    // Call backend to delete
    try {
      const chat = chatHistory.find((c) => c.id === id);
      if (chat) {
        await fetch(`${API}/session/${chat.sessionId}`, {
          method: "DELETE",
        });
      }
    } catch (err) {
      console.error("Failed to delete chat:", err);
    }
  };

  // ── Reset pipeline visual ───────────────────────────────────────────────
  const resetPipeline = () => {
    setActiveStep(null);
    setDoneSteps([]);
    setWebFallback(undefined);
    setCurrentStepLabel("");
  };

  // ── Send message via SSE ────────────────────────────────────────────────
  const sendMessage = async (e) => {
    e?.preventDefault();
    const msg = input.trim();
    if (!msg || loading || waitingForFeedback) return;

    setInput("");
    setLoading(true);
    resetPipeline();

    const userMsg = { role: "user", content: msg };
    setMessages((prev) => [...prev, userMsg]);

    // Update chat title on first message
    setMessages((prev) => {
      if (prev.length === 1) {
        setChatHistory((chats) =>
          chats.map((c) =>
            c.id === activeChatId
              ? { ...c, title: msg.length > 28 ? msg.slice(0, 28) + "…" : msg }
              : c
          )
        );
      }
      return prev;
    });

    // Use /stream-chat endpoint to get real-time pipeline events
    try {
      const res = await fetch(`${API}/stream-chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: msg }),
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let finalData = null;

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split("\n\n");
          for (const block of lines) {
            if (block.startsWith("data: ")) {
              try {
                const data = JSON.parse(block.substring(6));
                
                if (data.step === "final") {
                  finalData = data;
                } else if (data.step === "error") {
                  setMessages((prev) => [
                    ...prev,
                    { role: "ai", content: `⚠️ **Error**: ${data.detail}`, webFallback: false },
                  ]);
                } else {
                  // Real-time pipeline step update
                  setActiveStep(data.step);
                  setDoneSteps((prev) => [...new Set([...prev, data.step])]);
                  if (data.detail) {
                    setCurrentStepLabel(data.detail);
                  }
                }
              } catch (e) {}
            }
          }
        }
      }

      if (finalData) {
        setWebFallback(finalData.web_fallback);
        setActiveStep(null);
        setCurrentStepLabel("");
        
        setMessages((prev) => [
          ...prev,
          {
            role: "ai",
            content: finalData.content,
            webFallback: finalData.web_fallback,
            arcParams: finalData.arc_params,
          },
        ]);

        if (finalData.arc_params) setArcParams(finalData.arc_params);
        if (finalData.needs_feedback) {
          setWaitingForFeedback(true);
          setCurrentThread(sessionId);
        }
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "ai",
          content: "⚠️ **Connection Error** — Please ensure the backend is running on port 8000.",
          webFallback: false,
        },
      ]);
    } finally {
      setLoading(false);
      fetchStats();
      fetchSessions(); // refresh sidebar to show new sessions
    }
  };

  // ── Feedback ────────────────────────────────────────────────────────────
  const handleFeedback = async (isHelpful) => {
    setWaitingForFeedback(false);
    try {
      await fetch(`${API}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          thread_id: currentThread,
          is_helpful: isHelpful,
          comments: isHelpful ? "" : "User rejected the response.",
        }),
      });
      if (!isHelpful) {
        setMessages((prev) => [
          ...prev,
          {
            role: "ai",
            content:
              "Got it! The ARC has adjusted retrieval parameters — I'll do a broader search next time.",
            webFallback: false,
          },
        ]);
      }
    } catch (_) {}
  };

  // ── Textarea auto-resize ────────────────────────────────────────────────
  const handleInputChange = (e) => {
    setInput(e.target.value);
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // ── Chip click ──────────────────────────────────────────────────────────
  const handleChipClick = (q) => {
    setInput(q);
    setTimeout(() => sendMessage({ preventDefault: () => {} }), 50);
  };

  // ── Pipeline running status ─────────────────────────────────────────────
  const pipelineActive = loading;

  return (
    <>
      {showSplash && (
        <div className="splash-screen">
          <div className="splash-logo">⚡️</div>
          <div className="splash-title">ACARA</div>
        </div>
      )}
      <div className="app-shell" style={{ display: showSplash ? "none" : "flex" }}>
        {/* ── Sidebar ── */}
      <div className={`sidebar ${sidebarOpen ? "" : "collapsed"}`}>
        <div className="sidebar-inner">
          <div className="sidebar-header">
            <div className="sidebar-logo">⚡</div>
            <div>
              <div className="sidebar-title">ACARA</div>
              <div className="sidebar-subtitle">Adaptive Context-Aware</div>
            </div>
          </div>

          <button className="btn-new-chat" onClick={handleNewChat}>
            <span>＋</span> New Chat
          </button>

          <div className="sidebar-section-label">Recent Chats</div>

          <div className="chat-history-list">
            {chatHistory.map((chat) => (
              <div
                key={chat.id}
                className={`chat-history-item ${activeChatId === chat.id ? "active" : ""}`}
                onClick={() => handleSwitchChat(chat.id)}
                style={{ display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer", textAlign: "left" }}
              >
                <div style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  💬 {chat.title}
                </div>
                <button
                  onClick={(e) => handleDeleteChat(e, chat.id)}
                  title="Delete chat"
                  style={{
                    background: "transparent", border: "none", color: "#ef4444",
                    cursor: "pointer", padding: "0 4px", fontSize: "14px",
                    opacity: activeChatId === chat.id ? 1 : 0.6
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.opacity = 1)}
                  onMouseLeave={(e) => (e.currentTarget.style.opacity = activeChatId === chat.id ? 1 : 0.6)}
                >
                  ✕
                </button>
              </div>
            ))}
          </div>

          <div className="sidebar-stats">
            <div className="stats-row">
              <span>Vector Memory</span>
              <span className="stats-val">{vectorDocCount} docs</span>
            </div>
            <div className="stats-row">
              <span>Session</span>
              <span className="stats-val">{sessionId}</span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Main Content ── */}
      <div className="main-content">
        {/* Topbar */}
        <div className="topbar">
          <button className="btn-icon" onClick={() => setSidebarOpen((p) => !p)} title="Toggle sidebar">
            {sidebarOpen ? "◀" : "▶"}
          </button>
          <span className="topbar-title">Adaptive Context-Aware RAG</span>

          <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
            <div className="pipeline-badge">
              <div className={`pipeline-badge-dot ${pipelineActive ? "active" : ""}`} />
              {pipelineActive ? (currentStepLabel || "Processing…") : "Ready"}
            </div>
            <button
              className="btn-icon"
              onClick={() => setShowUploadModal(true)}
              title="Inject document into Vector Memory"
              style={{ fontSize: 16 }}
            >
              📄
            </button>
            <button
              className="btn-icon"
              onClick={() => setRightPanelOpen((p) => !p)}
              title="Toggle pipeline panel"
            >
              {rightPanelOpen ? "▶" : "◀"}
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="body-panel">
          {/* Chat Panel */}
          <div className="chat-panel">
            <div className="messages-container">
              {historyLoading ? (
                <div className="empty-state">
                  <div className="typing-indicator" style={{ justifyContent: "center" }}>
                    <div className="typing-dot" /><div className="typing-dot" /><div className="typing-dot" />
                    <span className="step-label">Loading history…</span>
                  </div>
                </div>
              ) : messages.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-logo">⚡️</div>
                  <h1 className="empty-title">How can I help you today?</h1>
                  
                  <div className="empty-chips">
                    <div className="chip" onClick={() => { setInput("What is the score of Machine Learning in SHL?"); setTimeout(() => textareaRef.current?.focus(), 10); }}>
                      <span className="chip-title">Find a specific fact</span>
                      <span className="chip-desc">Ask about scores or data from your PDFs</span>
                    </div>
                    <div className="chip" onClick={() => { setInput("Provide a brief summary of the most recently uploaded document"); setTimeout(() => textareaRef.current?.focus(), 10); }}>
                      <span className="chip-title">Summarize documents</span>
                      <span className="chip-desc">Get a high-level overview of injected knowledge</span>
                    </div>
                    <div className="chip" onClick={() => { setInput("Explain how the Adaptive RAG pipeline works"); setTimeout(() => textareaRef.current?.focus(), 10); }}>
                      <span className="chip-title">Understand the Agent</span>
                      <span className="chip-desc">Learn about the internal LangGraph architecture</span>
                    </div>
                    <div className="chip" onClick={() => { setInput("Search the web for the latest news on Agentic RAG"); setTimeout(() => textareaRef.current?.focus(), 10); }}>
                      <span className="chip-title">Trigger Web Search</span>
                      <span className="chip-desc">Force the agent to fallback to external knowledge</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="messages-inner">
                  {messages.map((msg, idx) => (
                    <div key={idx} className={`message-row ${msg.role}`}>
                      <div className={`avatar ${msg.role === "user" ? "user" : "ai"}`}>
                        {msg.role === "user" ? "U" : "AI"}
                      </div>
                      <div className={`message-bubble ${msg.role === "user" ? "user" : "ai"}`}>
                        {msg.role === "ai" && msg.webFallback !== undefined && (
                          <div className={`route-badge ${msg.webFallback ? "web" : "memory"}`}>
                            {msg.webFallback ? "🌐 Web Search Path" : "💾 Vector Memory Path"}
                          </div>
                        )}
                        {msg.role === "ai" ? (
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {msg.content}
                          </ReactMarkdown>
                        ) : (
                          msg.content
                        )}
                      </div>
                    </div>
                  ))}

                  {/* Loading indicator */}
                  {loading && (
                    <div className="message-row ai">
                      <div className="avatar ai">AI</div>
                      <div className="typing-indicator">
                        <div className="typing-dot" />
                        <div className="typing-dot" />
                        <div className="typing-dot" />
                        {currentStepLabel && (
                          <span className="step-label">{currentStepLabel}…</span>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Feedback */}
                  {waitingForFeedback && !loading && (
                    <div className="message-row ai">
                      <div className="avatar ai" style={{ opacity: 0 }} />
                      <div className="feedback-card">
                        <p className="feedback-question">
                          Did this answer your question?
                        </p>
                        <div className="feedback-btns">
                          <button
                            className="btn-feedback yes"
                            onClick={() => handleFeedback(true)}
                          >
                            👍 Yes, helpful
                          </button>
                          <button
                            className="btn-feedback no"
                            onClick={() => handleFeedback(false)}
                          >
                            👎 No, search again
                          </button>
                        </div>
                      </div>
                    </div>
                  )}

                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {/* Input Area */}
            <div className="input-area">
              <div className="input-wrapper">
                <div className="input-row">
                  <button
                    className="btn-upload-inline"
                    onClick={() => setShowUploadModal(true)}
                    title="Inject document into Vector Memory"
                  >
                    📎
                  </button>
                  <div className="input-box">
                    <textarea
                      ref={textareaRef}
                      className="input-textarea"
                      value={input}
                      onChange={handleInputChange}
                      onKeyDown={handleKeyDown}
                      disabled={loading || waitingForFeedback}
                      placeholder={
                        waitingForFeedback
                          ? "Please provide feedback before continuing…"
                          : "Ask anything — Shift+Enter for new line…"
                      }
                      rows={1}
                    />
                  </div>
                  <button
                    className="btn-send"
                    onClick={sendMessage}
                    disabled={!input.trim() || loading || waitingForFeedback}
                  >
                    ↑
                  </button>
                </div>
                <p className="input-hint">
                  Adaptive RAG · Dynamic Chunking · Credibility Scoring · ARC
                </p>
              </div>
            </div>
          </div>

          {/* ── Right Panel ── */}
          <div className={`right-panel ${rightPanelOpen ? "" : "collapsed"}`}>
            <div className="right-panel-inner">
              {/* Pipeline Visualizer */}
              <div className="panel-section-title">Live Pipeline</div>
              <PipelineVisualizer
                activeStep={activeStep}
                doneSteps={doneSteps}
                webFallback={webFallback}
              />

              {/* ARC Panel */}
              <div className="panel-section-title" style={{ marginTop: 8 }}>
                ARC Parameters
              </div>
              <ARCPanel externalParams={arcParams} />

              {/* Upload Panel */}
              <div className="panel-section-title" style={{ marginTop: 8 }}>
                Inject Knowledge
              </div>
              <UploadPanel
                onUploaded={() => {
                  fetchStats();
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* ── Upload Modal ── */}
      {showUploadModal && (
        <div className="modal-overlay" onClick={() => setShowUploadModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <span className="modal-title">📄 Inject into Vector Memory</span>
              <button className="btn-modal-close" onClick={() => setShowUploadModal(false)}>
                ✕
              </button>
            </div>
            <UploadPanel
              onUploaded={() => {
                fetchStats();
                setTimeout(() => setShowUploadModal(false), 1500);
              }}
            />
          </div>
        </div>
      )}
      </div>
    </>
  );
}

export default App;
