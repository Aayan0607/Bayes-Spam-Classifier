"""Phase 1: an explainable multinomial Naive Bayes spam classifier.

Uses only Python's standard library. The model is implemented from scratch so
that priors, conditional probabilities, Laplace smoothing and Bayes scoring
remain visible for the Discrete Mathematics project.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
STOP_WORDS = set("a an the and or but if as at by for from in into of on to with is am are was were be been being i me my we us our you your he him his she her it its they them their this that these those do does did have has had not no so than then there here".split())


def tokenize(message: str) -> list[str]:
    """Convert a message into a discrete sequence of lowercase word tokens."""
    return TOKEN_PATTERN.findall(message.lower())


class NaiveBayesSpamClassifier:
    """Multinomial Naive Bayes with add-one Laplace smoothing."""

    def __init__(self) -> None:
        self.class_message_counts: Counter[str] = Counter()
        self.word_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.class_word_totals: Counter[str] = Counter()
        self.vocabulary: set[str] = set()
        self.total_messages = 0
        self.temperature = 1.0

    def train(self, examples: list[tuple[str, str]], sample_weights=None) -> None:
        if not examples:
            raise ValueError("Training data cannot be empty.")

        self.__init__()
        document_counts = Counter()
        for _, message in examples:
            document_counts.update(set(tokenize(message)) - STOP_WORDS)
        # Use training documents only. Rare words give unreliable likelihoods.
        self.vocabulary = {word for word, count in document_counts.items() if count >= 3}
        if not self.vocabulary or {label for label, _ in examples} != {'spam', 'ham'}:
            raise ValueError('Training requires both classes and a nonempty vocabulary.')

        weights = sample_weights if sample_weights is not None else [1] * len(examples)
        if len(weights) != len(examples) or any(w <= 0 for w in weights):
            raise ValueError('Each training example requires one positive weight.')
        for (label, message), weight in zip(examples, weights):
            if label not in {"spam", "ham"}:
                raise ValueError(f"Unsupported label: {label!r}")
            words = [w for w in tokenize(message) if w in self.vocabulary]
            self.class_message_counts[label] += weight
            self.word_counts[label].update({w: n * weight for w, n in Counter(words).items()})
            self.class_word_totals[label] += len(words) * weight

        self.total_messages = sum(weights)

    def prior(self, label: str) -> float:
        return self.class_message_counts[label] / self.total_messages

    def word_probability(self, word: str, label: str) -> float | None:
        """Return P(word | label) using Laplace smoothing."""
        if word not in self.vocabulary:
            return None
        numerator = self.word_counts[label][word] + 1
        denominator = self.class_word_totals[label] + len(self.vocabulary)
        return numerator / denominator

    def token_reason(self, word: str) -> str:
        if word in STOP_WORDS:
            return 'common word ignored'
        if word not in self.vocabulary:
            return 'unseen or rare word ignored'
        return 'used'

    def word_log_probability(self, word: str, label: str) -> float:
        return math.log(self.word_probability(word, label)) if word in self.vocabulary else 0.0

    def log_score(self, message: str, label: str) -> float:
        """Compute log P(label) + sum(log P(word | label))."""
        score = math.log(self.prior(label))
        for word in tokenize(message):
            score += self.word_log_probability(word, label)
        return score

    def predict(self, message: str) -> dict:
        """Predict a class and return the mathematics used in the decision."""
        scores = {label: self.log_score(message, label) for label in ("spam", "ham")}

        # Stable two-class softmax converts log scores into comparable values.
        maximum = max(scores.values())
        weights = {label: math.exp((score - maximum) / self.temperature) for label, score in scores.items()}
        total = sum(weights.values())
        probabilities = {label: weight / total for label, weight in weights.items()}
        class_prediction = max(probabilities, key=probabilities.get)
        prediction = 'review' if 0.4 < probabilities['spam'] < 0.6 else class_prediction

        evidence = []
        for word in sorted(set(tokenize(message))):
            spam_probability = self.word_probability(word, "spam")
            ham_probability = self.word_probability(word, "ham")
            influence = self.word_log_probability(word, 'spam') - self.word_log_probability(word, 'ham')
            evidence.append(
                {
                    "word": word,
                    "spam_probability": spam_probability,
                    "ham_probability": ham_probability,
                    "influence": influence,
                    "reason": self.token_reason(word),
                }
            )
        evidence.sort(key=lambda item: abs(item["influence"]), reverse=True)

        return {
            "prediction": prediction,
            "class_prediction": class_prediction,
            "probabilities": probabilities,
            "log_scores": scores,
            "priors": {label: self.prior(label) for label in ("spam", "ham")},
            "evidence": evidence,
            "temperature": self.temperature,
        }


def load_dataset(path: Path) -> list[tuple[str, str]]:
    csv.field_size_limit(10_000_000)
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return [(row["label"].strip().lower(), row["message"].strip()) for row in rows]


def split_dataset(examples: list[tuple[str, str]], test_ratio: float = 0.25):
    """Make a deterministic stratified split so both classes occur in each set."""
    random_generator = random.Random(42)
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for example in examples:
        groups[example[0]].append(example)

    training, testing = [], []
    for group in groups.values():
        random_generator.shuffle(group)
        test_size = max(1, round(len(group) * test_ratio))
        testing.extend(group[:test_size])
        training.extend(group[test_size:])
    random_generator.shuffle(training)
    random_generator.shuffle(testing)
    return training, testing


def build_model():
    """Keep the public test set untouched; domain examples are training-only.

    Temperature scaling is fitted on an inner calibration split, never on the
    public test set. It reduces overconfidence without changing score ordering.
    """
    root = Path(__file__).resolve().parent
    training, testing = split_dataset(load_dataset(root / 'data.csv'))
    supplemental = load_dataset(root / 'transaction_examples.csv')
    fitting, calibration = split_dataset(training, 0.2)
    model = NaiveBayesSpamClassifier()
    # Domain adaptation: the small authored collection gets explicit weight,
    # rather than being swamped by thousands of historical mailing-list emails.
    domain_weight = 20
    model.train(fitting + supplemental, [1] * len(fitting) + [domain_weight] * len(supplemental))
    margins = [(model.log_score(text, 'spam') - model.log_score(text, 'ham'), label == 'spam') for label, text in calibration]
    def loss(temperature):
        total = 0.0
        for margin, spam in margins:
            value = margin / temperature
            total += max(value, 0) - value * spam + math.log1p(math.exp(-abs(value)))
        return total / len(margins)
    temperature = min((1, 2, 3, 5, 8, 12, 20, 30), key=loss)
    model.train(training + supplemental, [1] * len(training) + [domain_weight] * len(supplemental))
    model.temperature = temperature
    return model, training, testing


def evaluate(model: NaiveBayesSpamClassifier, testing: list[tuple[str, str]]) -> dict:
    counts = Counter()
    review_counts = Counter()
    for actual, message in testing:
        scores = {label: model.log_score(message, label) for label in ('spam', 'ham')}
        predicted = max(scores, key=scores.get)
        if abs((scores['spam'] - scores['ham']) / model.temperature) < math.log(1.5):
            review_counts[actual] += 1
        if actual == "spam" and predicted == "spam":
            counts["tp"] += 1
        elif actual == "ham" and predicted == "spam":
            counts["fp"] += 1
        elif actual == "ham" and predicted == "ham":
            counts["tn"] += 1
        else:
            counts["fn"] += 1

    total = len(testing)
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    accuracy = (counts["tp"] + counts["tn"]) / total if total else 0
    return {
        "tp": counts["tp"],
        "fp": counts["fp"],
        "tn": counts["tn"],
        "fn": counts["fn"],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "review_spam": review_counts['spam'],
        "review_ham": review_counts['ham'],
    }


def print_explanation(message: str, result: dict) -> None:
    print(f"\nMessage: {message}")
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Adjusted spam score: {result['probabilities']['spam'] * 100:.2f}%")
    print(f"Adjusted ham score: {result['probabilities']['ham'] * 100:.2f}%")
    print("\nBayes components")
    for label in ("spam", "ham"):
        print(
            f"  {label:4}  prior={result['priors'][label]:.4f}  "
            f"log score={result['log_scores'][label]:.4f}  "
            f"normalized={result['probabilities'][label]:.4f}"
        )

    print("\nStrongest word evidence")
    print("  word             P(w|spam)   P(w|ham)    log ratio   direction")
    for item in result["evidence"][:8]:
        direction = "spam" if item["influence"] > 0 else "ham" if item["influence"] < 0 else "neutral"
        print(
            f"  {item['word'][:15]:15}  "
            f"{format(item['spam_probability'], '.5f') if item['spam_probability'] is not None else 'ignored':<11} "
            f"{format(item['ham_probability'], '.5f') if item['ham_probability'] is not None else 'ignored':<11} "
            f"{item['influence']:<11.4f} {direction}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Explainable Bayes spam classifier")
    parser.add_argument("message", nargs="*", help="Message to classify")
    parser.add_argument("--evaluate", action="store_true", help="Show test-set metrics")
    args = parser.parse_args()

    model, training, testing = build_model()

    if args.evaluate:
        metrics = evaluate(model, testing)
        print(f"Training messages: {len(training)} | Test messages: {len(testing)}")
        print(
            f"Confusion matrix: TP={metrics['tp']} FP={metrics['fp']} "
            f"TN={metrics['tn']} FN={metrics['fn']}"
        )
        print(
            f"Accuracy={metrics['accuracy']:.3f}  Precision={metrics['precision']:.3f}  "
            f"Recall={metrics['recall']:.3f}  F1={metrics['f1']:.3f}"
        )
        print(f"Binary benchmark above. Review policy: {metrics['review_ham']} ham and {metrics['review_spam']} spam messages require review.")

    message = " ".join(args.message).strip()
    if message:
        print_explanation(message, model.predict(message))
    elif not args.evaluate:
        print("Enter a message to classify. Press Enter on an empty line to quit.")
        while True:
            message = input("\nMessage: ").strip()
            if not message:
                break
            print_explanation(message, model.predict(message))


if __name__ == "__main__":
    main()
