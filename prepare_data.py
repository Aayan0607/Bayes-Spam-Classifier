"""Rebuild data.csv from Apache SpamAssassin's public email corpus."""
import csv
import html
import io
import re
import tarfile
import urllib.request
from email import policy
from email.parser import BytesParser
from pathlib import Path

BASE = 'https://spamassassin.apache.org/old/publiccorpus/'
ARCHIVES = [('20030228_easy_ham.tar.bz2', 'ham'), ('20030228_easy_ham_2.tar.bz2', 'ham'),
            ('20030228_hard_ham.tar.bz2', 'ham'), ('20030228_spam.tar.bz2', 'spam'),
            ('20050311_spam_2.tar.bz2', 'spam')]


def visible_text(message):
    parts = list(message.walk()) if message.is_multipart() else [message]
    plain, markup = [], []
    for part in parts:
        if part.get_content_disposition() == 'attachment':
            continue
        kind = part.get_content_type()
        if kind not in {'text/plain', 'text/html'}:
            continue
        try:
            text = part.get_content()
        except (LookupError, UnicodeError):
            text = (part.get_payload(decode=True) or b'').decode('utf-8', errors='replace')
        if not isinstance(text, str):
            continue
        if kind == 'text/html':
            text = re.sub(r'<(script|style)\b[^>]*>.*?</\1>', ' ', text, flags=re.S | re.I)
            markup.append(html.unescape(re.sub(r'<[^>]+>', ' ', text)))
        else:
            plain.append(text)
    return str(message.get('Subject', '')) + ' ' + ' '.join(plain or markup)


def main():
    records = {}
    for archive, label in ARCHIVES:
        print('Downloading', archive, flush=True)
        with urllib.request.urlopen(BASE + archive, timeout=90) as response:
            blob = response.read()
        with tarfile.open(fileobj=io.BytesIO(blob), mode='r:bz2') as source:
            for member in source:
                if not member.isfile() or member.name.endswith('cmds'):
                    continue
                message = BytesParser(policy=policy.default).parsebytes(source.extractfile(member).read())
                text = re.sub(r'\s+', ' ', visible_text(message)).strip()
                key = ' '.join(re.findall(r"[a-z0-9']+", text.lower()))
                if len(key) < 20:
                    continue
                # Drop conflicting labels rather than choosing one arbitrarily.
                if key in records and records[key][0] != label:
                    records[key] = ('conflict', text)
                else:
                    records.setdefault(key, (label, text))
    output = Path(__file__).with_name('data.csv')
    with output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['label', 'message'])
        writer.writerows(row for row in records.values() if row[0] != 'conflict')
    print('Saved', sum(row[0] != 'conflict' for row in records.values()), 'unique emails to', output)


if __name__ == '__main__':
    main()
