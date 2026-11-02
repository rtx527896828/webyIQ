// The toolbar button opens the side panel, where the model and the agent run.
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {});

// Automated tests (tools/gpu-bench.ps1 -Extension): a tab opened on the local dev server at
// /ext-autotest?... is turned into the panel page in test mode with the same query.
function toTestPanel(tabId, href) {
  const url = href && new URL(href);
  if (url && url.origin === "http://127.0.0.1:8765" && url.pathname === "/ext-autotest") {
    chrome.tabs.update(tabId, { url: chrome.runtime.getURL("sidepanel.html") + url.search });
  }
}
chrome.tabs.onUpdated.addListener((tabId, change) => toTestPanel(tabId, change.url));
// (the tab may have been opened before this worker started)
chrome.tabs.query({ url: "http://127.0.0.1:8765/ext-autotest*" }).then((tabs) => tabs.forEach((t) => toTestPanel(t.id, t.url)));
