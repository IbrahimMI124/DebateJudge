import requests
from bs4 import BeautifulSoup
import json
import re
import time

players = {
    "Messi": "https://en.wikipedia.org/wiki/Lionel_Messi",
    "Ronaldo": "https://en.wikipedia.org/wiki/Cristiano_Ronaldo",
    "Neymar": "https://en.wikipedia.org/wiki/Neymar",
    "Mbappe": "https://en.wikipedia.org/wiki/Kylian_Mbapp%C3%A9",
    "Neuer": "https://en.wikipedia.org/wiki/Manuel_Neuer",
    "Buffon": "https://en.wikipedia.org/wiki/Gianluigi_Buffon",
    "Haaland": "https://en.wikipedia.org/wiki/Erling_Haaland",
    "Zidane": "https://en.wikipedia.org/wiki/Zinedine_Zidane",
    "Ronaldinho": "https://en.wikipedia.org/wiki/Ronaldinho"
}

headers = {
    "User-Agent": "Mozilla/5.0"
}

corpus = []

def clean(txt):
    txt = re.sub(r'\[[0-9]+\]', '', txt)   # remove citations
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

for name, url in players.items():
    try:
        print("Scraping", name)
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "lxml")

        paragraphs = soup.find_all("p")

        count = 0
        for p in paragraphs:
            text = clean(p.get_text())

            if len(text) > 80:
                corpus.append({
                    "source": "Wikipedia",
                    "url": url,
                    "entity": name,
                    "text": text
                })
                count += 1

            if count >= 8:
                break

        time.sleep(1)

    except Exception as e:
        print("Error:", name, e)

with open("data/corpus.json", "w", encoding="utf-8") as f:
    json.dump(corpus, f, indent=2, ensure_ascii=False)

print("Saved", len(corpus), "entries")