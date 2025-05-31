
# CP423 – Assignment 1  

# Imports

import queue
import re
from collections import defaultdict
from urllib.parse import urljoin, urlparse

import nltk
import requests
from bs4 import BeautifulSoup


# Constants

SEED_URL = "https://en.wikipedia.org/wiki/Canada"
WIKI_ROOT = "https://en.wikipedia.org"
MAX_DEPTH = 2          # link depth
MAX_PAGES = 200        # absolute page limit
HEADERS = {"User-Agent": "CP423-A1 student crawler"}

# stop-word list and simple tokeniser (no downloads required)
nltk_stop = set(nltk.corpus.stopwords.words("english"))
token_rx = re.compile(r"[a-z0-9]+")


# Crawl

def normalise(url: str) -> str:
    url = url.split("#")[0]     # This function cleans up the text by removing slashes and converting to lower case 
    url = url.rstrip("/")       # to help with duplicates
    return url.lower()          

def valid_article(url: str) -> bool:    #this removes un-needed pages like help
    pr = urlparse(url)
    bad = (":", "/help:", "/file:", "/special:")
    return pr.netloc.endswith("wikipedia.org") and not any(x in pr.path.lower() for x in bad)

def extract_text(soup: BeautifulSoup) -> str:        #this cleans up the text more
    for x in soup.select(
        "#mw-navigation, #footer, table.infobox, table.navbox, div.reflist"
    ):
        x.decompose()
    body = soup.select_one("div.mw-parser-output")
    return body.get_text(" ", strip=True) if body else ""

def crawl(seed: str = SEED_URL, depth_limit: int = MAX_DEPTH) -> dict:     #Breadth-first crawl up to layer 2 
    pages, visited = {}, set()
    q = queue.Queue()
    q.put((seed, 0))

    while not q.empty() and len(visited) < MAX_PAGES:
        url, depth = q.get()
        url = normalise(url)

        if depth > depth_limit or url in visited:
            continue

        print(f"[{len(visited):3}] fetching {url}")

        try:
            html = requests.get(url, headers=HEADERS, timeout=5).text
        except Exception as err:
            print("skip", url, "reason:", err)
            continue

        visited.add(url)
        soup = BeautifulSoup(html, "lxml")
        pages[url] = extract_text(soup)

        if depth < depth_limit:
            for tag in soup.select("a[href^='/wiki/']"):
                link = urljoin(WIKI_ROOT, tag["href"])
                if valid_article(link):
                    q.put((link, depth + 1))

    print("crawl finished:", len(visited), "pages")
    return pages


# Index and preprocessing

def preprocess(text: str) -> list[str]:   #tokenizes and makes lowercase and also drops un-needed words
    return [
        t for t in token_rx.findall(text.lower())
        if len(t) > 1 and t not in nltk_stop
    ]

def build_index(corpus: dict) -> dict:    #this returns the inverted index
    index = defaultdict(set)
    for url, text in corpus.items():
        for term in preprocess(text):
            index[term].add(url)
    return index


# Boolean search

class BooleanSearch:
    def __init__(self, inv: dict):
        self.inv = inv

    def _tokens(self, query: str) -> list[str]:                   #allows AND OR NOT boolean operations
        spaced = query.upper().replace("(", " ( ").replace(")", " ) ").split()
        out = []
        for part in spaced:
            if part in ("AND", "OR", "NOT", "(", ")"):
                out.append(part)
            else:
                tok = preprocess(part)
                if tok:
                    out.append(tok[0])
        return out

    def _to_rpn(self, query: str) -> list[str]:       #Uses RPN
        prec = {"NOT": 3, "AND": 2, "OR": 1}
        out, ops = [], []
        for tok in self._tokens(query):
            if tok in prec:
                while ops and ops[-1] != "(" and prec[ops[-1]] >= prec[tok]:
                    out.append(ops.pop())
                ops.append(tok)
            elif tok == "(":
                ops.append(tok)
            elif tok == ")":
                while ops and ops[-1] != "(":
                    out.append(ops.pop())
                if ops:
                    ops.pop()
            else:
                out.append(tok)
        while ops:
            out.append(ops.pop())
        return out

    def search(self, query: str) -> set[str]:    #this evaluates RPN with the set operations
        rpn, st = self._to_rpn(query), []
        for tok in rpn:
            if tok == "NOT":
                a = st.pop()
                universe = set().union(*self.inv.values()) if self.inv else set()
                st.append(universe - a)
            elif tok in ("AND", "OR"):
                b, a = st.pop(), st.pop()
                st.append(a & b if tok == "AND" else a | b)
            else:
                st.append(self.inv.get(tok, set()))
        return st.pop() if st else set()


# REPL

def repl(searcher: BooleanSearch) -> None:      #this is the comand interface for the queries
    print("\nBoolean query prompt (type quit to exit)\n")
    while True:
        q = input("query> ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        hits = searcher.search(q)
        print(len(hits), "documents matched")
        for url in list(hits)[:10]:
            print(" ", url)
        print()


# Main

def main() -> None:
    pages = crawl()
    index = build_index(pages)
    searcher = BooleanSearch(index)
    repl(searcher)

if __name__ == "__main__":
    main()
