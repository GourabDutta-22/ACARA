/**
 * config.js
 * ---------
 * Centralized configuration for the frontend.
 * Handles API URL discovery and environment variables.
 */

// 1. Try environment variable (baked in at build time)
let API_URL = import.meta.env.VITE_API_URL;

// 2. Smart Discovery for Render
// If we are on Render (*.onrender.com) and the API_URL wasn't baked in,
// we attempt to guess it based on the current hostname.
if (!API_URL && window.location.hostname.includes("onrender.com")) {
  // If frontend is 'acara-rag.onrender.com', we guess backend is 'acara-backend.onrender.com'
  // Or if it's a pull request preview, it might be more complex, but we try the most common pattern.
  const backendUrl = window.location.origin.replace("acara-rag", "acara-backend");
  // Only use it if it's different (don't point at yourself)
  if (backendUrl !== window.location.origin) {
    API_URL = backendUrl;
    console.log("⚡️ ACARA: Smart Discovery guessed Backend URL:", API_URL);
  }
}

// 3. Final Fallback (Local Development)
if (!API_URL) {
  API_URL = "http://localhost:8000";
}

// Ensure no trailing slash
export const API = API_URL.endsWith("/") ? API_URL.slice(0, -1) : API_URL;

console.table({
  "ACARA API CONFIG": {
    URL: API,
    Source: import.meta.env.VITE_API_URL ? "Environment Variable" : "Smart Discovery / Fallback",
    Environment: import.meta.env.MODE
  }
});
