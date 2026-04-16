/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** e.g. https://api.example.com — omit in dev to use the Vite proxy */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
