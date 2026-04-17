import { useCallback, useEffect, useState } from "react";

import "./App.css";
import {
  getHealth,
  getModelInfo,
  postChat,
  type Role,
} from "./api/client";

export default function App() {
  const [role, setRole] = useState<Role>("patient");
  const [context, setContext] = useState("");
  const [message, setMessage] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [modelLoaded, setModelLoaded] = useState<boolean | null>(null);
  const [mockGeneration, setMockGeneration] = useState<boolean | null>(null);

  const refreshBackend = useCallback(async () => {
    try {
      const h = await getHealth();
      setBackendOk(h.status === "ok");
      const m = await getModelInfo();
      setModelLoaded(Boolean(m.loaded));
      setMockGeneration(Boolean(m.mock_generation));
    } catch {
      setBackendOk(false);
      setModelLoaded(null);
      setMockGeneration(null);
    }
  }, []);

  useEffect(() => {
    void refreshBackend();
  }, [refreshBackend]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setAnswer(null);
    const q = message.trim();
    if (!q) {
      setError("Please enter a question.");
      return;
    }
    setLoading(true);
    try {
      const res = await postChat({
        role,
        message: q,
        context: context.trim() || null,
      });
      setAnswer(res.answer);
      void refreshBackend();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>CliniQ</h1>
        <p>Medical Q&A — pick an audience so the assistant can tune depth and wording.</p>
        <div
          className="backend-pill"
          data-ok={backendOk === true ? "true" : "false"}
          title="FastAPI /health and /api/v1/model"
        >
          <span className="dot" aria-hidden />
          {backendOk === null && "Checking backend…"}
          {backendOk === false && "Backend unreachable (start uvicorn on :8000)"}
          {backendOk === true && (
            <>
              API connected
              {mockGeneration === true && " · Mock replies (no LLM)"}
              {mockGeneration !== true && modelLoaded === false && " · Model not loaded (chat returns 503)"}
              {mockGeneration !== true && modelLoaded === true && " · Model ready"}
            </>
          )}
        </div>
      </header>

      <form onSubmit={onSubmit}>
        <fieldset className="role-fieldset">
          <legend>Audience</legend>
          <div className="role-toggle" role="radiogroup" aria-label="Audience">
            <label>
              <input
                type="radio"
                name="role"
                value="patient"
                checked={role === "patient"}
                onChange={() => {
                  setRole("patient");
                }}
              />
              <span className="role-card">
                <strong>Patient</strong>
                <span>Plain language, safety and when to seek care</span>
              </span>
            </label>
            <label>
              <input
                type="radio"
                name="role"
                value="doctor"
                checked={role === "doctor"}
                onChange={() => {
                  setRole("doctor");
                }}
              />
              <span className="role-card">
                <strong>Doctor</strong>
                <span>Terminology and clinically denser detail</span>
              </span>
            </label>
          </div>
        </fieldset>

        <div className="field">
          <label htmlFor="context">Optional context</label>
          <textarea
            id="context"
            value={context}
            onChange={(ev) => {
              setContext(ev.target.value);
            }}
            placeholder="History, meds, snippets from notes…"
            maxLength={16000}
          />
        </div>

        <div className="field">
          <label htmlFor="question">Question (required)</label>
          <textarea
            id="question"
            value={message}
            onChange={(ev) => {
              setMessage(ev.target.value);
            }}
            placeholder="Ask your medical question in one or more sentences…"
            maxLength={8000}
            required
          />
        </div>

        <div className="actions">
          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? "Generating…" : "Submit"}
          </button>
          <button
            type="button"
            className="btn-secondary"
            onClick={() => {
              void refreshBackend();
            }}
            disabled={loading}
          >
            Refresh status
          </button>
        </div>
      </form>

      {error ? <div className="alert">{error}</div> : null}

      {answer ? (
        <section className="answer-card" aria-live="polite">
          <h2>Answer</h2>
          <p className="answer-body">{answer}</p>
        </section>
      ) : null}
    </div>
  );
}
