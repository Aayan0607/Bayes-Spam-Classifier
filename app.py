"""Local Phase 2 interface. Run python app.py, then visit localhost:5000."""
import json
import math
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from classifier import build_model, evaluate, tokenize

ROOT = Path(__file__).resolve().parent
model, training, testing = build_model()
metrics = evaluate(model, testing)


def explain(message):
    result = model.predict(message)
    counts = Counter(tokenize(message))
    for item in result["evidence"]:
        word = item["word"]
        item["count"] = counts[word]
        item["contribution"] = counts[word] * item["influence"]
        item["known"] = word in model.vocabulary
        item["training_counts"] = {c: model.word_counts[c][word] for c in ("spam", "ham")}
    result["evidence"].sort(key=lambda item: abs(item["contribution"]), reverse=True)
    scores = {c: math.log(model.prior(c)) for c in ("spam", "ham")}
    prior_margin = (scores['spam'] - scores['ham']) / model.temperature
    steps = [{"word": "Prior", "spam_probability": 1 / (1 + math.exp(-prior_margin)), "log_odds": scores["spam"] - scores["ham"]}]
    for word in tokenize(message):
        for c in scores:
            scores[c] += model.word_log_probability(word, c)
        maximum = max(scores.values())
        spam, ham = (math.exp((scores[c] - maximum) / model.temperature) for c in ("spam", "ham"))
        steps.append({"word": word, "reason": model.token_reason(word), "spam_probability": spam / (spam + ham), "log_odds": scores["spam"] - scores["ham"]})
    result["steps"] = steps
    result["word_log_sums"] = {c: result["log_scores"][c] - math.log(model.prior(c)) for c in scores}
    result["word_totals"] = dict(model.class_word_totals)
    result["vocabulary_size"] = len(model.vocabulary)
    return result


class Handler(BaseHTTPRequestHandler):
    def send(self, status, data, content_type="application/json; charset=utf-8"):
        body = json.dumps(data).encode() if content_type.startswith("application/json") else data
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/model-info":
            self.send(200, {"training": len(training), "testing": len(testing), "vocabulary": len(model.vocabulary), "metrics": metrics})
        else:
            self.send(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/predict":
            return self.send(404, {"error": "Not found"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 100000:
                return self.send(413, {"error": "Enter a message smaller than 100 KB."})
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            subject, body = payload.get("subject", ""), payload.get("body", "")
            if not isinstance(subject, str) or not isinstance(body, str):
                raise ValueError("Subject and body must be text.")
            message = (subject + " " + body).strip()
            if not tokenize(message):
                raise ValueError("Enter a message containing words or numbers.")
            self.send(200, explain(message))
        except (ValueError, UnicodeError) as error:
            self.send(400, {"error": str(error)})


if __name__ == "__main__":
    print("BayesMail Inspector: http://127.0.0.1:5000  (Ctrl+C to stop)", flush=True)
    try:
        ThreadingHTTPServer(("127.0.0.1", 5000), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
