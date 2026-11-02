// Automated tests only (sidepanel.html?autotest): report page start and uncaught errors to the dev server.
if (location.search.includes("autotest")) {
  const log = (m) => fetch("http://127.0.0.1:8765/log?m=" + encodeURIComponent(m)).catch(() => {});
  log("panel page loaded");
  addEventListener("error", (e) => log(`error: ${e.message} at ${e.filename}:${e.lineno}`));
  addEventListener("unhandledrejection", (e) => log(`rejection: ${e.reason?.stack || e.reason}`));
}
