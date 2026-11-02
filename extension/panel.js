const $ = id => document.getElementById(id);
const percent = value => `${(value * 100).toFixed(2)}%`;
const number = value => value.toFixed(4);
let current = null, capture = null, requestId = 0, controller = null, consuming = false;

function clear(message) {
  requestId++;
  controller?.abort();
  current = null; capture = null;
  $('results').hidden = true; $('preview').hidden = true;
  $('body').textContent = ''; $('subject').textContent = '';
  $('words').replaceChildren(); $('likelihoods').replaceChildren();
  $('analyse').disabled = false;
  $('status').textContent = message;
  $('error').textContent = '';
}

function showStep() {
  if (!current) return;
  const index = Number($('step').value), step = current.steps[index];
  $('step-label').textContent = index ? `${index}/${current.steps.length - 1}: “${step.word}” (${step.reason})` : 'Starting point: adjusted class priors';
  $('step-bar').style.width = percent(step.spam_probability);
  $('step-value').textContent = `Spam ${percent(step.spam_probability)} · log odds ${number(step.log_odds)}`;
}
$('step').addEventListener('input', showStep);

function addRow(target, values, contribution) {
  const row = document.createElement('tr');
  values.forEach((value, index) => {
    const cell = document.createElement('td');
    cell.textContent = String(value);
    if (index === 2 && contribution) cell.className = contribution > 0 ? 'positive' : 'negative';
    row.append(cell);
  });
  $(target).append(row);
}

function render(result) {
  current = result;
  $('results').hidden = false;
  $('prediction').textContent = result.prediction === 'review' ? 'Uncertain — review this message' : result.prediction === 'spam' ? 'Likely spam' : 'Likely legitimate (ham)';
  $('probability').textContent = result.prediction === 'review' ? 'Mixed evidence. Check the sender and transaction context.' : `Adjusted model score: ${percent(result.probabilities[result.prediction])}`;
  $('spam').textContent = `Spam ${percent(result.probabilities.spam)}`;
  $('ham').textContent = `Ham ${percent(result.probabilities.ham)}`;
  $('probability-bar').style.width = percent(result.probabilities.spam);
  $('step').max = result.steps.length - 1; $('step').value = 0; showStep();
  $('words').replaceChildren(); $('likelihoods').replaceChildren();
  for (const item of result.evidence) {
    addRow('words', [item.word + (item.reason === 'used' ? '' : ` (${item.reason})`), item.count, number(item.contribution)], item.contribution);
    addRow('likelihoods', [item.word, item.spam_probability?.toFixed(6) ?? 'ignored', item.ham_probability?.toFixed(6) ?? 'ignored']);
  }
  $('scores').replaceChildren();
  for (const label of ['spam', 'ham']) {
    const p = document.createElement('p'); p.className = 'formula';
    p.textContent = `${label}: ln(${number(result.priors[label])}) + (${number(result.word_log_sums[label])}) = ${number(result.log_scores[label])}`;
    $('scores').append(p);
  }
  const max = Math.max(...Object.values(result.log_scores));
  const temperature = result.temperature || 1;
  const a = Math.exp((result.log_scores.spam - max) / temperature), b = Math.exp((result.log_scores.ham - max) / temperature);
  $('normalization').textContent = `Adjusted spam score = exp((score_spam − max)/T) / Σ exp((score_C − max)/T), T=${temperature}: ${a.toPrecision(5)} / (${a.toPrecision(5)} + ${b.toPrecision(5)}) = ${percent(result.probabilities.spam)}. Temperature reduces overconfidence; word contributions and log odds remain raw Bayes values. Scores between 40% and 60% require review.`;
  $('vocabulary').textContent = `Vocabulary: ${result.vocabulary_size} words. Training word totals: spam ${result.word_totals.spam}, ham ${result.word_totals.ham}.`;
  const prior = Math.log(result.priors.spam / result.priors.ham);
  const sum = result.evidence.reduce((total, item) => total + item.contribution, 0);
  $('reconciliation').textContent = `Prior log odds (${number(prior)}) + word evidence (${number(sum)}) = final log odds (${number(result.log_scores.spam - result.log_scores.ham)})`;
}

async function analyse(data, id) {
  if (id !== requestId) return;
  const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
  if (id !== requestId) return;
  if (tab?.id !== data.tabId || tab.url !== data.url) throw Error('The source page changed. Capture the current message again.');
  if (!data.body.trim()) throw Error('The selected text is empty.');
  const payload = JSON.stringify({subject: data.subject, body: data.body});
  if (new TextEncoder().encode(payload).length > 100000) throw Error('This message is too large (100 KB limit). Select a shorter passage and use the right-click menu.');
  capture = data;
  $('preview').hidden = false;
  $('subject').textContent = data.subject || 'Selected text';
  $('body').textContent = data.body;
  $('source').textContent = `${data.source}${data.expandedMessages ? ` (${data.expandedMessages} expanded in this conversation)` : ''} · ${new Date().toLocaleTimeString()}`;
  $('status').textContent = 'Captured. Analysing with your local Python server…';
  controller = new AbortController();
  const activeController = controller;
  const timer = setTimeout(() => activeController.abort(), 20000);
  try {
    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: payload, signal: controller.signal
    });
    const result = await response.json();
    if (id !== requestId) return;
    if (!response.ok) throw Error(result.error || 'The local server rejected this message.');
    if (!result.steps || !result.evidence) throw Error('Restart the updated CP/app.py server and try again.');
    render(result);
    $('status').textContent = 'Analysis complete. Results apply to the captured text below.';
  } catch (error) {
    if (id !== requestId) return;
    if (error instanceof TypeError || error.name === 'AbortError') throw Error('Cannot reach the local classifier. Run python app.py in CP, wait for the localhost address, then try again.');
    throw error;
  } finally { clearTimeout(timer); }
}

async function run(getData) {
  clear('Reading the current message…');
  const id = requestId;
  $('analyse').disabled = true;
  try { await analyse(await getData(), id); }
  catch (error) { if (id === requestId) { $('error').textContent = error.message; $('status').textContent = 'Analysis did not complete.'; } }
  finally { if (id === requestId) $('analyse').disabled = false; }
}
$('analyse').addEventListener('click', () => run(async () => {
  const response = await chrome.runtime.sendMessage({type: 'extract-gmail'});
  if (!response?.ok) throw Error(response?.error || 'Could not read this page.');
  return response.data;
}));

async function consumePending() {
  if (consuming) return;
  consuming = true;
  try {
  const {pending} = await chrome.storage.session.get('pending');
  if (!pending) return;
  const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
  if (tab?.id !== pending.tabId) return;
  await chrome.storage.session.remove('pending');
  if (Date.now() - pending.time > 60000) return;
  await run(async () => pending);
  } finally { consuming = false; }
}
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === 'session' && changes.pending?.newValue) consumePending().catch(console.error);
});
chrome.tabs.onActivated.addListener(() => clear('Tab changed. Capture the message you want to analyse.'));
chrome.tabs.onUpdated.addListener((tabId, change) => {
  if (capture?.tabId === tabId && (change.url || change.status === 'loading')) clear('Page changed. Analyse the newly opened message.');
});
consumePending().catch(error => { $('error').textContent = error.message; });
