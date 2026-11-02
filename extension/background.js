// No automatic scanning. Browser actions open the panel; page reads need a click.
chrome.sidePanel.setPanelBehavior({openPanelOnActionClick: true}).catch(console.error);
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => chrome.contextMenus.create({
    id: 'analyse-selection', title: 'Analyse selected text with BayesMail',
    contexts: ['selection'], documentUrlPatterns: ['http://*/*', 'https://*/*']
  }));
});
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId !== 'analyse-selection' || !tab?.id) return;
  // open() must run directly in the user gesture, before asynchronous storage.
  const opened = chrome.sidePanel.open({windowId: tab.windowId});
  const pending = {tabId: tab.id, url: tab.url, subject: '', body: info.selectionText || '', source: 'Selected text', time: Date.now()};
  opened.then(() => chrome.storage.session.set({pending})).catch(console.error);
});

// Serialized by scripting.executeScript. It has no access to extension state.
function extractGmail() {
  const visible = element => {
    const style = getComputedStyle(element);
    return element.getClientRects().length > 0 && style.visibility !== 'hidden' && style.display !== 'none';
  };
  if (location.hostname !== 'mail.google.com') throw Error('Open Gmail, or select text and use the right-click menu.');
  const root = [...document.querySelectorAll('[role="main"]')].find(visible) || document;
  const bodies = [...root.querySelectorAll('.a3s')].filter(visible).filter(el => !el.closest('[role="dialog"], [contenteditable="true"]'));
  if (!bodies.length) throw Error('Open an email and expand its message body first. Inbox previews and compose windows cannot be analysed.');
  // Gmail renders expanded message bodies in conversation order.
  const last = bodies[bodies.length - 1];
  const body = last.innerText.trim();
  if (!body) throw Error('The expanded email has no readable text. Images and attachments need a separate OCR feature.');
  const subject = [...root.querySelectorAll('h2.hP, h2[data-thread-perm-id]')].find(visible)?.innerText.trim() || '';
  return {subject, body, source: 'Latest expanded Gmail message', url: location.href, expandedMessages: bodies.length};
}

chrome.runtime.onMessage.addListener((message, sender, reply) => {
  if (sender.id !== chrome.runtime.id || sender.tab || message.type !== 'extract-gmail') return;
  (async () => {
    const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
    if (!tab?.id || !tab.url?.startsWith('https://mail.google.com/')) throw Error('Open an email in Gmail. On another website, highlight text and use the right-click menu.');
    const results = await chrome.scripting.executeScript({target: {tabId: tab.id}, func: extractGmail});
    const data = results[0]?.result;
    if (!data?.body) throw Error('Could not read this message. Refresh Gmail and try again.');
    const after = await chrome.tabs.get(tab.id);
    if (after.url !== tab.url) throw Error('The page changed during extraction. Try again on the opened email.');
    reply({ok: true, data: {...data, tabId: tab.id}});
  })().catch(error => reply({ok: false, error: error.message}));
  return true;
});
