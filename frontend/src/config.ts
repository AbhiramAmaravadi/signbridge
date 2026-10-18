/**
 * Central API configuration for the SignBridge frontend.
 * Override with VITE_API_BASE_URL in `.env` (see `.env.example`).
 */
const DEFAULT_API_BASE_URL =
  'https://signbridge-backend-389644353290.us-central1.run.app';

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL
).replace(/\/$/, '');

/** Shared JSON headers for all backend requests. */
export const JSON_HEADERS = {
  'Content-Type': 'application/json',
  Accept: 'application/json',
} as const;

/** Inference / prediction */
export const INFERENCE_URL = `${API_BASE_URL}/inference`;

/** Sentence assembly + Gemini translation (same Cloud Run service) */
export const FINALIZE_URL = `${API_BASE_URL}/api/v1/sentence/finalize`;
export const RESET_URL = `${API_BASE_URL}/api/v1/sentence/reset`;
export const APPEND_WORD_URL = `${API_BASE_URL}/api/v1/sentence/append`;
export const DELETE_WORD_URL = `${API_BASE_URL}/api/v1/sentence/delete`;
export const TRANSLATE_URL = `${API_BASE_URL}/api/v1/gemini/translate`;

/** Short host label for connection-status UI */
export const API_HOST_LABEL = (() => {
  try {
    return new URL(API_BASE_URL).host;
  } catch {
    return API_BASE_URL;
  }
})();

/** OpenAPI / Swagger docs on Cloud Run */
export const API_DOCS_URL = `${API_BASE_URL}/docs`;

/** Backend health endpoint (also used to wake a scaled-to-zero Cloud Run instance). */
export const HEALTH_URL = `${API_BASE_URL}/health`;

/**
 * Fire-and-forget request sent as soon as the page loads so a cold Cloud Run
 * instance starts booting (model + MediaPipe init) before the user clicks "Start".
 */
export function warmUpBackend(): void {
  try {
    void fetch(HEALTH_URL, { method: 'GET', mode: 'cors', cache: 'no-store' }).catch(() => undefined);
  } catch {
    /* ignore: warm-up is best effort */
  }
}
