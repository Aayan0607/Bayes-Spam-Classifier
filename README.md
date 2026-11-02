# BayesMail Inspector

The project is divided into three deliberately small phases.

## Phase 1 — Mathematical classifier (implemented)

- Multinomial Naive Bayes written from scratch in Python
- Class priors and conditional word probabilities
- Add-one Laplace smoothing for unseen words
- Log probabilities for numerical stability
- Normalized spam and ham scores
- Word-level influence using `log(P(word|spam) / P(word|ham))`
- Confusion matrix, accuracy, precision, recall and F1 score

Run it with Python 3:

```powershell
python classifier.py --evaluate
python classifier.py "WIN a free prize now"
python classifier.py "The college is offering a free workshop"
```

Running `python classifier.py` without a message starts interactive mode.

The CSV contains 5,907 unique emails from the historical [Apache SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/). `prepare_data.py` downloads the easy ham, easy ham 2, hard ham, spam and spam 2 collections and rebuilds this file with Python's standard library. It reads subject and visible message body, skips attachments and mail-routing headers, prefers plain text over duplicate HTML alternatives, and removes normalized duplicates and conflicting labels before splitting. Existing messages are replaced when you explicitly run `python prepare_data.py`.

The model ignores a fixed list of common connecting words (including `and`) and retains words appearing in at least three training documents. Words outside this training vocabulary contribute zero. Laplace smoothing applies only to retained vocabulary words; a retained word absent from one class still receives a nonzero probability in that class. The API marks ignored tokens and displays their likelihood as null rather than inventing a vocabulary probability.

### Evaluation recorded 26 October 2026

Seed 42, stratified 75/25 split: 4,430 training messages and 1,477 held-out messages. All vocabulary selection uses training data only. Training: 3,081 ham and 1,349 spam messages, vocabulary: 18,156 words.

| Metric | Result |
|---|---:|
| Accuracy | 97.90% |
| Spam precision | 98.38% |
| Spam recall | 94.67% |
| Spam F1 | 96.49% |
| True positives | 426 |
| False positives | 7 |
| True negatives | 1,020 |
| False negatives | 24 |

These are results for the final configuration on this historical corpus, not an accuracy comparison against the previous 40-example demo. Extra manually written checks correctly classified a project email, college workshop message and two promotional spam messages. A legitimate appointment reminder still produced a false positive (81.63% model spam probability). No checks were added to the training set. Modern notification and transactional emails need better coverage before claiming reliable Gmail performance.

Mathematical checks confirmed that retained-word likelihoods sum to one, ignored tokens contribute zero, repeated-word contributions reconstruct final log odds, the slider ends at the same probability as the full prediction, and normalized duplicates do not cross the split. Intermediate slider scores can still rise and fall; they describe only the prefix examined so far.

## Phase 2 — Explainable local interface (implemented)

### Billing false-positive update

The old public corpus incorrectly gave strong spam contributions to ordinary billing language. Four manually written receipts/statements/invoice/refund probes originally scored 99.40% to nearly 100% spam. This update adds `transaction_examples.csv`: 40 explicitly authored legitimate billing examples and 12 authored payment scams, used only for training. They are illustrative domain-adaptation examples, not independent real-world evaluation data. Each receives training weight 20 to give the small contemporary example set a meaningful contribution alongside thousands of historical emails. Weighted word counts and class priors are shown transparently in the explanation. The public `data.csv` and its outer test split remain unchanged. Run the CLI and app through `build_model()` to use this configuration.

Temperature scaling reduces overconfidence: an inner 20% split of the public training partition selects temperature by log loss from a fixed grid. The final model refits the outer training partition plus weighted examples and applies the selected temperature (currently 20). The outer test set does not select temperature. Raw Bayes log scores remain available; displayed adjusted probabilities and the slider both use `exp((score - max)/temperature)`. Refitting and domain shift mean these adjusted scores still do not establish reliability on modern Gmail.

The result is now `spam`, `ham`, or `review`. Spam scores strictly between 40% and 60% return **Uncertain — review this message**. The 40/60 band is a conservative product policy, not an empirically optimal threshold. `class_prediction` preserves the raw binary choice for inspection. Do not treat `review` as verified ham.

Updated historical binary benchmark: TP 425, FP 6, TN 1,021, FN 25 (97.90% accuracy, 98.61% precision, 94.44% recall). Under the review policy, 41/1,477 messages require review (32 spam, 9 ham). Decided messages: 405 spam correctly flagged, 1,012 ham correctly accepted, 6 false spam warnings and 13 spam accepted as ham. Binary metrics shown in the interface include review cases via their binary score choice and must not be confused with the review-policy counts.

Four out-of-training billing smoke checks now return review with spam scores 46.24%–54.63%, rather than confident spam. A lottery processing-fee scam returns spam (68.62%); a credential-stealing refund request requires review (59.81%). These small authored probes guided this fix and are not a blind test or an accuracy claim. Mathematical reconciliation and slider-final-score checks passed.

After updating: stop and restart `python app.py`, Reload the extension at `chrome://extensions` or `edge://extensions`, close/reopen its panel, then analyse again. The extension UI must be reloaded to support the new review state.

Run `python app.py` from this folder and open http://127.0.0.1:5000.

Only two files are added: `app.py` and `index.html`. Python's built-in HTTP server replaces Flask so no packages or installation are needed. Stop with Ctrl+C. This is a local demonstration server.

The page accepts a subject and message, shows normalized probabilities, and explains the log scores, Laplace-smoothed word probabilities and repeated-word contributions. A slider shows how the probability changes as each token contributes evidence. Evaluation uses the same deterministic 75/25 split as the CLI.

API for Phase 3: `POST /predict` with JSON `{"subject": "...", "body": "..."}`. `GET /model-info` returns dataset sizes and held-out metrics. Messages stay on the local machine and are not saved by the application.

### Remaining model follow-up

- Common-word bias and out-of-vocabulary scoring are corrected as described above. Extend the historical corpus with representative modern email data, particularly legitimate reminders and account notifications, using a separate validation set for future model decisions.
- Keep explanations phrased as model contributions, not statements that a word is inherently suspicious.
- Check calibration before describing normalized model probabilities as reliable confidence estimates.

## Phase 3 — Gmail extension (implemented, live Gmail check pending)

The `extension` folder contains five plain files: `manifest.json`, `background.js`, `panel.html`, `panel.js`, and `panel.css`. No build step or packages are needed. Use Chrome 116+ or an Edge version supporting the Chrome side panel API.

### Install and use

1. In this folder run `python app.py`. Wait for `http://127.0.0.1:5000` to appear. Leave that terminal running.
2. Open `chrome://extensions` (or `edge://extensions`) and enable Developer mode.
3. Click **Load unpacked** and select `C:\Users\ADMIN\Desktop\SY\DM\CP\extension` (the folder containing the manifest).
4. Pin BayesMail Inspector from the browser's extensions menu.
5. Open an email in Gmail and expand the message body. Click the extension icon, then **Analyse this email** in its side panel.
6. Review **Captured text** to confirm the intended subject/body, then inspect the prediction, slider, word contributions and expandable Bayes calculation.

For text on another webpage, highlight it and right-click **Analyse selected text with BayesMail**. This opens the panel and analyses that selection. It also works for selecting a particular passage within Gmail.

Gmail extraction targets the last expanded message body in conversation order, not necessarily the latest message if later replies are collapsed. Expand the intended last reply or select its text explicitly. Visible quoted history and signatures inside the message body can be included; the preview shows exactly what was analysed. Inbox lists, compose windows, images, scanned documents and desktop apps are not supported. Gmail's DOM can change; the extraction selectors may need maintenance.

Reads occur only after clicking Analyse or the selection menu. The extension does not label, move or alter mail. Prediction requests go from the extension panel to the fixed local endpoint, using extension host permissions rather than opening the Python API to all websites. Gmail access is limited to `mail.google.com`; `activeTab` supports user-invoked access on other pages. Selected text waits briefly in browser session storage and is removed when consumed. Results otherwise remain in panel memory and clear on tab/page changes. No email text is written to disk by the extension.

### Troubleshooting and checking on your browser

- **Cannot reach the local classifier:** start or restart `python app.py`, then retry. No CORS installation or Node server is required.
- **No readable email:** open an email, expand its body, and click Analyse again. Use selected text if Gmail's layout differs.
- **Changes to extension files:** click Reload on its browser extensions page, then close/reopen the panel. Refresh Gmail if needed.
- Confirm that opening the panel alone sends no text. Analyse a known email and review the extraction preview. Compare its result to the same subject/body in the local webpage. Move the slider to its last step and confirm it matches the result. Navigate to another message and confirm the old result clears.
- With the Python server stopped, check that the panel gives a restart instruction. On an ordinary webpage, verify the selection menu analyses only the selected text. Restricted browser pages such as `chrome://` cannot be read.

Implementation references: [Chrome Side Panel API](https://developer.chrome.com/docs/extensions/reference/api/sidePanel), [scripting API](https://developer.chrome.com/docs/extensions/reference/api/scripting), and [extension network requests](https://developer.chrome.com/docs/extensions/develop/concepts/network-requests). Loading into a signed-in browser and confirming current Gmail selectors requires a live browser check; static and simulated checks cannot establish Gmail compatibility.
