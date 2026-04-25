/**
 * HTTP client for CliniQ FastAPI — matches
 * backend/app/schemas/chat.py (ChatRequest / ChatResponse).
 */

export type Role = "doctor" | "patient";

export interface ChatRequestBody {
  role: Role;
  message: string;
  context?: string | null;
}

export interface ChatResponseBody {
  answer: string;
  role: Role;
}

function apiBase(): string {
  const raw = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
  return raw.replace(/\/$/, "");
}

function joinUrl(path: string): string {
  const base = apiBase();
  if (!base) {
    return path.startsWith("/") ? path : `/${path}`;
  }
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
}

function readDetail(payload: unknown): string {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const d = (payload as { detail: unknown }).detail;
    if (typeof d === "string") {
      return d;
    }
    if (Array.isArray(d)) {
      return d.map((x) => JSON.stringify(x)).join("; ");
    }
    return JSON.stringify(d);
  }
  return "";
}

export async function getHealth(): Promise<{ status: string }> {
  const res = await fetch(joinUrl("/health"));
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<{ status: string }>;
}

export async function getModelInfo(): Promise<Record<string, unknown>> {
  const res = await fetch(joinUrl("/api/v1/model"));
  if (!res.ok) {
    throw new Error(`Model info failed: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<Record<string, unknown>>;
}

export async function postChat(body: ChatRequestBody): Promise<ChatResponseBody> {
  const payload = {
    role: body.role,
    message: body.message.trim(),
    context: body.context?.trim() ? body.context.trim() : null,
  };

  const res = await fetch(joinUrl("/api/v1/chat"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const text = await res.text();
  let json: unknown = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    json = null;
  }

  if (!res.ok) {
    const detail = readDetail(json) || text || res.statusText;
    throw new Error(detail);
  }

  return json as ChatResponseBody;
}
