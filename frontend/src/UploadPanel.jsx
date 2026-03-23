/**
 * UploadPanel
 * ===========
 * Drag-and-drop + textarea input to manually inject documents
 * into the Vector Memory via the /upload endpoint.
 */

import { useState, useRef } from "react";

import { API } from "./config";

function UploadPanel({ onUploaded }) {
  const [text, setText] = useState("");
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState(null); // { type: 'success'|'error', msg }
  const fileInputRef = useRef();

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };
  const handleDragLeave = () => setIsDragOver(false);

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = [...e.dataTransfer.files];
    for (const file of files) {
      if (file.type === "application/pdf" || file.name.endsWith(".pdf")) {
        await uploadPdf(file);
      } else if (file.type === "text/plain" || file.name.endsWith(".txt") || file.name.endsWith(".md")) {
        const content = await file.text();
        setText((prev) => prev ? prev + "\n\n" + content : content);
      }
    }
  };

  const handleFileSelect = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    if (file.type === "application/pdf" || file.name.endsWith(".pdf")) {
      await uploadPdf(file);
    } else {
      const content = await file.text();
      setText((prev) => prev ? prev + "\n\n" + content : content);
    }
    e.target.value = "";
  };

  const uploadPdf = async (file) => {
    setUploading(true);
    setStatus(null);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API}/upload-pdf`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      
      if (res.ok) {
        setStatus({
          type: "success",
          msg: `✓ ${data.chunks_stored} chunk(s) from "${data.filename}" (${data.pages_extracted} pages) stored`,
        });
        onUploaded?.();
      } else {
        setStatus({ type: "error", msg: `Error: ${data.detail || "PDF upload failed"}` });
      }
    } catch (err) {
      setStatus({ type: "error", msg: `Connection error: ${err.message}` });
    } finally {
      setUploading(false);
    }
  };

  const handleUpload = async () => {
    if (!text.trim()) return;
    setUploading(true);
    setStatus(null);
    try {
      const res = await fetch(`${API}/upload`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text.trim(),
          metadata: { source: "manual_upload", uploaded_at: new Date().toISOString() },
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setStatus({
          type: "success",
          msg: `✓ ${data.chunks_stored} chunk(s) stored in Vector Memory`,
        });
        setText("");
        onUploaded?.();
      } else {
        setStatus({ type: "error", msg: `Error: ${data.detail || "Upload failed"}` });
      }
    } catch (err) {
      setStatus({ type: "error", msg: `Connection error: ${err.message}` });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-panel">
      {/* Dropzone */}
      <div
        className={`upload-dropzone ${isDragOver ? "drag-over" : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <span className="upload-dropzone-icon">📄</span>
        <p className="upload-dropzone-label">
          <strong>Drop a .txt, .md, or .pdf file</strong>
          <br />
          or click to browse
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept=".txt,.md,.pdf"
          style={{ display: "none" }}
          onChange={handleFileSelect}
        />
      </div>

      {/* Text input */}
      <textarea
        className="upload-textarea"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Or paste text directly here…"
        rows={4}
      />

      <button
        className="btn-upload-submit"
        onClick={handleUpload}
        disabled={!text.trim() || uploading}
      >
        {uploading ? "Storing in Vector Memory…" : "⬆  Inject into Vector Memory"}
      </button>

      {status && (
        <div className={`upload-status ${status.type}`}>
          {status.msg}
        </div>
      )}
    </div>
  );
}

export default UploadPanel;
