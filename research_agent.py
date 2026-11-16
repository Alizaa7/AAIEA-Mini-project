"""
Autonomous Research & Intelligence Agent  (shared by Colab notebook + Streamlit app)

Agents (all coordinated by ResearchAgent = the Orchestrator):
 1 QueryUnderstanding  2 Planner  3 SourceDiscovery  4 DocumentReader
 5 CredibilityAnalyzer 6 FactExtractor  7 CrossVerifier  8 KnowledgeGraph
 9 ReportWriter  10 CitationManager  + Memory, Dynamic Expansion, Interactive mode

LLM: Groq (free API key -> https://console.groq.com/keys). Set env var GROQ_API_KEY.
Search: DuckDuckGo (no key) + arXiv API (no key).
"""
import os, re, json, time, hashlib, datetime, urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable

import requests
import networkx as nx

import logging
for _n in ("trafilatura", "pypdf", "httpx", "urllib3"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)   # silence harmless parser warnings

UA = {"User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"}
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
MEMORY_FILE = os.environ.get("RESEARCH_MEMORY", "research_memory.json")


# ----------------------------------------------------------------------------- helpers
def safe_json(txt: str) -> dict:
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def domain_of(url: str) -> str:
    h = urllib.parse.urlparse(url).netloc.lower()
    return h[4:] if h.startswith("www.") else h


def sid_of(url: str) -> str:
    return "S" + hashlib.md5(url.encode()).hexdigest()[:6]


def tokens(text: str) -> set:
    stop = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "for", "on", "with",
            "how", "what", "will", "be", "by", "as", "at", "it", "this", "that", "from", "over", "next"}
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in stop and len(w) > 2}


# ----------------------------------------------------------------------------- LLM wrapper
CACHE_FILE = os.environ.get("RESEARCH_LLM_CACHE", "llm_cache.json")
# when a model hits its DAILY token quota, continue on the next one (each model has its own quota on Groq)
DEFAULT_FALLBACKS = ["openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


class LLM:
    """Groq wrapper: disk cache (identical prompt = 0 API calls), call counter, smart 429 handling."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL, cache_file: Optional[str] = CACHE_FILE,
                 fallbacks: Optional[List[str]] = None):
        from groq import Groq
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY missing. Set it in Colab secrets / env / Streamlit sidebar.")
        self.client = Groq(api_key=key)
        self.model = model
        self.calls, self.hits = 0, 0
        self.fallbacks = [m for m in (DEFAULT_FALLBACKS if fallbacks is None else fallbacks) if m != model]
        self.dead = set()          # models whose daily quota is exhausted
        self.cache_file = cache_file
        try:
            self.cache = json.load(open(cache_file)) if cache_file else {}
        except Exception:
            self.cache = {}

    def _save(self):
        if self.cache_file:
            try:
                json.dump(self.cache, open(self.cache_file, "w"))
            except Exception:
                pass

    @staticmethod
    def _wait(err, attempt):
        m = re.search(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)", str(err))
        if m:
            secs = int(m.group(1) or 0) * 60 + float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
            return min(secs + 1, 70)
        return min(2 ** attempt * 2, 40)

    def _alive(self, m):
        if m not in self.dead:
            return m
        return next((c for c in [self.model] + self.fallbacks if c not in self.dead), m)

    def chat(self, system: str, user: str, json_mode=False, temperature=0.2, max_tokens=2500, model=None,
             effort="low") -> str:
        model = model or self.model
        last = None
        for attempt in range(8):
            model = self._alive(model)
            key = hashlib.sha1(json.dumps([model, system, user, json_mode, max_tokens]).encode()).hexdigest()
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            try:
                kw = {"response_format": {"type": "json_object"}} if json_mode else {}
                mt = max_tokens
                if "gpt-oss" in model:
                    kw["reasoning_effort"] = effort      # low = far fewer hidden thinking tokens
                    mt = max_tokens * 2                  # headroom for the remaining reasoning tokens
                self.calls += 1
                r = self.client.chat.completions.create(
                    model=model, temperature=temperature, max_tokens=mt,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
                out = r.choices[0].message.content
                if not out:
                    raise RuntimeError("empty completion")
                self.cache[key] = out
                self._save()
                return out
            except Exception as e:
                last = e
                low = str(e).lower()
                if "per day" in low or "tpd" in low or "rpd" in low:     # daily quota: switch model instead of retrying
                    self.dead.add(model)
                    nxt = next((c for c in [self.model] + self.fallbacks if c not in self.dead), None)
                    if nxt is None:
                        raise RuntimeError(f"Daily Groq quota exhausted on all models ({[model] + self.fallbacks}). "
                                           f"Wait for reset (see error) or upgrade. Last error: {e}")
                    print(f"[LLM] daily limit hit on {model} -> switching to {nxt}")
                    if model == self.model:
                        self.model = nxt
                    model = nxt
                    continue
                time.sleep(self._wait(e, attempt))
        raise RuntimeError(f"LLM call failed: {last}")

    def json(self, task: str, system: str, user: str, **kw) -> dict:
        txt = self.chat(system + "\nReturn ONLY a valid JSON object.", user, json_mode=True, **kw)
        return safe_json(txt)


# ----------------------------------------------------------------------------- data models
@dataclass
class Source:
    url: str
    title: str = ""
    snippet: str = ""
    origin: str = "web"           # web | news | arxiv
    published: Optional[str] = None
    text: str = ""
    full_text: bool = False
    credibility: float = 0.0
    cred_breakdown: Dict = field(default_factory=dict)
    section: str = ""
    id: str = ""

    def __post_init__(self):
        self.id = self.id or sid_of(self.url)


@dataclass
class Fact:
    id: str
    source_id: str
    text: str
    type: str = "fact"            # fact | statistic | claim
    section: str = ""


@dataclass
class Session:
    query: str
    understanding: Dict = field(default_factory=dict)
    plan: List[Dict] = field(default_factory=list)
    sources: Dict[str, Source] = field(default_factory=dict)
    rejected: List[Source] = field(default_factory=list)
    facts: List[Fact] = field(default_factory=list)
    agreements: List[Dict] = field(default_factory=list)
    contradictions: List[Dict] = field(default_factory=list)
    graph: Optional[nx.DiGraph] = None
    report: str = ""
    sections_md: Dict[str, str] = field(default_factory=dict)
    refs: Dict[str, int] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- Agent 3: source discovery
def search_web(q: str, n=6) -> List[Source]:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        return [Source(url=r["href"], title=r.get("title", ""), snippet=r.get("body", ""), origin="web")
                for r in DDGS().text(q, max_results=n) if r.get("href")]
    except Exception:
        return []


def search_news(q: str, n=4) -> List[Source]:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        return [Source(url=r["url"], title=r.get("title", ""), snippet=r.get("body", ""), origin="news",
                       published=r.get("date")) for r in DDGS().news(q, max_results=n) if r.get("url")]
    except Exception:
        return []


def search_arxiv(q: str, n=4) -> List[Source]:
    try:
        words = list(tokens(q))[:6]
        if not words:
            return []
        r = requests.get("https://export.arxiv.org/api/query", headers=UA, timeout=25,
                         params={"search_query": " AND ".join(f"all:{w}" for w in words),
                                 "max_results": n, "sortBy": "relevance"})
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out = []
        for e in ET.fromstring(r.content).findall("a:entry", ns):
            link = e.find("a:id", ns).text.strip()
            title = " ".join(e.find("a:title", ns).text.split())
            summ = " ".join(e.find("a:summary", ns).text.split())
            pub = e.find("a:published", ns).text[:10]
            out.append(Source(url=link, title=title, snippet=summ[:300], origin="arxiv", published=pub,
                              text=f"{title}. {summ}", full_text=False))
        return out
    except Exception:
        return []


# ----------------------------------------------------------------------------- Agent 4: document reader
def read_url(src: Source, max_chars=6000) -> Source:
    if src.text:                         # e.g. arXiv abstract already present
        src.text = src.text[:max_chars]
        return src
    try:
        r = requests.get(src.url, headers=UA, timeout=20)
        ctype = r.headers.get("content-type", "").lower()
        if "pdf" in ctype or src.url.lower().endswith(".pdf"):
            import io
            from pypdf import PdfReader
            rd = PdfReader(io.BytesIO(r.content))
            txt = " ".join((p.extract_text() or "") for p in rd.pages[:8])
        else:
            import trafilatura
            txt = trafilatura.extract(r.text, include_comments=False, include_tables=False) or ""
            if not src.published:
                try:
                    md = trafilatura.extract_metadata(r.text)
                    if md and md.date:
                        src.published = md.date
                except Exception:
                    pass
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt) > 300:
            src.text, src.full_text = txt[:max_chars], True
            return src
    except Exception:
        pass
    src.text = src.snippet            # fall back to the search snippet
    src.full_text = False
    return src


# ----------------------------------------------------------------------------- Agent 5: credibility
GOV = ["nist.gov", "nsa.gov", "cisa.gov", "whitehouse.gov", "europa.eu", "enisa.europa.eu", "ncsc.gov.uk",
       "energy.gov", "nasa.gov", "who.int", "un.org", "oecd.org", "worldbank.org", "imf.org", "bis.org"]
JOURNAL = ["nature.com", "science.org", "ieee.org", "acm.org", "springer.com", "sciencedirect.com", "cell.com",
           "pnas.org", "aps.org", "nejm.org", "thelancet.com", "wiley.com", "arxiv-peer"]
NEWS = ["reuters.com", "ft.com", "bloomberg.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com", "wsj.com",
        "economist.com", "technologyreview.com", "quantamagazine.org", "wired.com", "arstechnica.com", "theguardian.com"]
INDUSTRY = ["ibm.com", "microsoft.com", "google.com", "mckinsey.com", "gartner.com", "deloitte.com", "bcg.com",
            "accenture.com", "forrester.com", "cloudflare.com", "nvidia.com", "aws.amazon.com", "cisco.com"]
LOWQ = ["medium.com", "substack.com", "blogspot.com", "wordpress.com", "quora.com", "reddit.com", "linkedin.com",
        "facebook.com", "twitter.com", "x.com", "tumblr.com", "pinterest.com", "youtube.com", "tiktok.com"]


def _match(d, lst):
    return any(d == x or d.endswith("." + x) for x in lst)


def score_credibility(src: Source) -> Source:
    """Transparent heuristic: domain authority + peer review + recency + content depth."""
    d = domain_of(src.url)
    peer = False
    if _match(d, GOV):
        base, kind = 95, "government / standards body"
    elif _match(d, JOURNAL):
        base, kind, peer = 92, "peer-reviewed journal / publisher", True
    elif d.endswith((".gov", ".edu", ".int")) or ".ac." in d or d.endswith(".gov.uk"):
        base, kind = 88, "government / academic institution"
    elif d.endswith("arxiv.org"):
        base, kind = 72, "preprint (not peer reviewed)"
    elif _match(d, NEWS):
        base, kind = 82, "established news outlet"
    elif _match(d, INDUSTRY):
        base, kind = 76, "industry / research firm"
    elif d.endswith("wikipedia.org"):
        base, kind = 58, "encyclopedia (secondary)"
    elif _match(d, LOWQ):
        base, kind = 30, "blog / social / user-generated"
    elif d.endswith(".org"):
        base, kind = 58, "organisation (.org)"
    else:
        base, kind = 48, "unknown domain"
    adj = {}
    if src.url.startswith("https"):
        adj["https"] = 2
    year = None
    if src.published:
        m = re.search(r"(19|20)\d{2}", str(src.published))
        year = int(m.group(0)) if m else None
    if year:
        age = datetime.date.today().year - year
        adj["recency"] = 3 if age <= 2 else (0 if age <= 6 else -8)
    if src.full_text:
        adj["full_text_read"] = 3
    elif len(src.text) < 200:
        adj["thin_content"] = -10
    score = max(5, min(99, base + sum(adj.values())))
    src.credibility = float(score)
    src.cred_breakdown = {"domain": d, "category": kind, "base": base, "peer_reviewed": peer, "adjustments": adj}
    return src


# ----------------------------------------------------------------------------- Orchestrator
class ResearchAgent:
    def __init__(self, llm=None, max_sources=16, min_credibility=40, max_contradictions=3,
                 expand_topics=True, fast_model=None, batch_size=3, memory_file=MEMORY_FILE, search_fn=None, news_fn=None, arxiv_fn=None,
                 reader_fn=None, log_fn: Callable[[str], None] = print):
        self.llm = llm or LLM()
        self.max_sources, self.min_cred = max_sources, min_credibility
        self.max_contra, self.expand_topics = max_contradictions, expand_topics
        self.fast_model, self.batch_size = fast_model, max(1, batch_size)   # optional cheap model for extraction only
        self.memory_file = memory_file
        self.search_web = search_fn or search_web
        self.search_news = news_fn or search_news
        self.search_arxiv = arxiv_fn or search_arxiv
        self.reader = reader_fn or read_url
        self._log = log_fn
        self.memory = self._load_memory()
        self.s: Optional[Session] = None

    # ---- logging / memory
    def log(self, msg):
        if self.s:
            self.s.log.append(msg)
        self._log(msg)

    def _load_memory(self):
        try:
            return json.load(open(self.memory_file))
        except Exception:
            return {"investigations": [], "url_cache": {}}

    def _save_memory(self):
        s = self.s
        self.memory["investigations"].append({
            "query": s.query, "date": str(datetime.date.today()),
            "topic": s.understanding.get("main_topic", ""), "n_sources": len(s.sources), "n_facts": len(s.facts)})
        for src in s.sources.values():
            self.memory["url_cache"][src.url] = {
                "title": src.title, "credibility": src.credibility, "published": src.published,
                "facts": [f.text for f in s.facts if f.source_id == src.id][:8]}
        try:
            json.dump(self.memory, open(self.memory_file, "w"), indent=1)
        except Exception:
            pass

    def related_past(self, query):
        qt = tokens(query)
        out = []
        for inv in self.memory.get("investigations", []):
            it = tokens(inv["query"] + " " + inv.get("topic", ""))
            if qt and len(qt & it) / len(qt | it) > 0.25:
                out.append(inv)
        return out

    # ---- main entry
    def run(self, query: str) -> Session:
        self.s = s = Session(query=query)
        c0, h0 = getattr(self.llm, "calls", 0), getattr(self.llm, "hits", 0)
        past = self.related_past(query)
        if past:
            self.log(f"[Memory] {len(past)} related past investigation(s) found; cached URLs will be reused.")
        self.understand(); self.plan()
        self.discover(); self.read_and_score(list(s.sources.values()))
        self.extract_all()
        if self.expand_topics:
            self.expand()
        self.verify()
        self.build_graph()
        self.write_report()
        self._save_memory()
        self.log(f"[Stats] Groq requests this run: {getattr(self.llm, 'calls', 0) - c0} | served from cache: {getattr(self.llm, 'hits', 0) - h0}")
        self.log("[Done] Report generated.")
        return s

    # ---- Agent 1: query understanding
    def understand(self):
        self.log("[1/9] Query Understanding Agent")
        self.s.understanding = self.llm.json(
            "understand",
            "You are a research query analyst. Output keys: main_topic (str), related_topics (list of str), "
            "research_objectives (list of 4-6 str), time_horizon (str or null), question_type (str).",
            self.s.query)

    # ---- Agent 2: planner
    def plan(self):
        self.log("[2/9] Research Planning Agent")
        u = self.s.understanding
        d = self.llm.json(
            "plan",
            "You are a research planner. Create 5-6 report sections that together answer the question. "
            "Output: {\"sections\":[{\"title\":str,\"goal\":str,\"queries\":[2 short web search queries]}]}. "
            "Make queries specific and aimed at diverse source types (academic, government/standards, industry, news).",
            f"Question: {self.s.query}\nUnderstanding: {json.dumps(u)}")
        secs = [x for x in d.get("sections", []) if x.get("title")][:6]
        if not secs:  # fallback
            secs = [{"title": o, "goal": o, "queries": [f"{self.s.query} {o}"]}
                    for o in (u.get("research_objectives") or [self.s.query])[:5]]
        self.s.plan = secs
        for x in secs:
            self.log(f"      - {x['title']}")

    # ---- Agent 3: discovery
    def _search_for(self, queries: List[str], section: str, per_q=5) -> List[Source]:
        found = []
        for i, q in enumerate(queries[:2]):
            found += self.search_web(q, per_q)
            if i == 0:
                found += self.search_arxiv(q, 3)
                found += self.search_news(q, 3)
        for f in found:
            f.section = section
        return found

    def discover(self):
        self.log("[3/9] Source Discovery Agent")
        s, per_section = self.s, max(2, self.max_sources // max(1, len(self.s.plan)))
        for sec in s.plan:
            added = 0
            for src in self._search_for(sec["queries"], sec["title"]):
                if src.id in s.sources or len(s.sources) >= self.max_sources + 4:
                    continue
                s.sources[src.id] = src
                added += 1
                if added >= per_section + 1:
                    break
        self.log(f"      {len(s.sources)} candidate sources")

    # ---- Agent 4 + 5: read, score, filter
    def read_and_score(self, sources: List[Source], grow=False):
        self.log("[4-5/9] Document Reader + Credibility Analyzer")
        with ThreadPoolExecutor(6) as ex:
            list(ex.map(self.reader, sources))
        for src in sources:
            score_credibility(src)
        for src in sources:
            if src.credibility < self.min_cred or len(src.text) < 80:
                self.s.rejected.append(src)
                self.s.sources.pop(src.id, None)
                self.log(f"      filtered: {domain_of(src.url)} (cred {src.credibility:.0f})")
        ranked = sorted(self.s.sources.values(), key=lambda x: -x.credibility)
        for extra in ([] if grow else ranked[self.max_sources:]):
            self.s.sources.pop(extra.id, None)
        self.log(f"      kept {len(self.s.sources)} sources")

    # ---- Agent 6: extraction (BATCHED: several documents per LLM call)
    def _add_items(self, items, src, titles):
        out = []
        for it in items[:6]:
            txt = (it.get("text") or "").strip()
            if len(txt) < 20:
                continue
            sec = it.get("section") if it.get("section") in titles else (src.section if src.section in titles else titles[0])
            typ = it.get("type") if it.get("type") in ("fact", "statistic", "claim") else "fact"
            f = Fact(id=f"F{len(self.s.facts) + 1}", source_id=src.id, text=txt, type=typ, section=sec)
            self.s.facts.append(f)
            out.append(f)
        return out

    def extract_many(self, srcs: List[Source], focus: str = "") -> List[Fact]:
        titles = [p["title"] for p in self.s.plan]
        out, todo = [], []
        for src in srcs:
            cached = self.memory.get("url_cache", {}).get(src.url)
            if cached and cached.get("facts"):                      # memory hit: no LLM call
                out += self._add_items([{"type": "fact", "text": x} for x in cached["facts"]], src, titles)
            elif len(src.text) >= 250:                              # skip near-empty pages: not worth a call
                todo.append(src)
        sysmsg = ("You extract structured research evidence. For EACH document list up to 5 items that help answer the "
                  "question. Item: {\"source\":the document id from its === header,\"type\":\"fact|statistic|claim\","
                  "\"text\":one self-contained sentence (keep numbers, dates, names),\"section\":one of the given section "
                  "titles}. Use 'claim' for predictions/opinions/forecasts. Never invent anything not in the text. "
                  "Output {\"items\":[...]}.")
        for i in range(0, len(todo), self.batch_size):
            chunk = todo[i:i + self.batch_size]
            per = 4500 if len(chunk) == 1 else 3800
            docs = "\n\n".join(f"=== {x.id} | {x.title} ===\n{x.text[:per]}" for x in chunk)
            d = self.llm.json("extract", sysmsg,
                              f"Question: {self.s.query}\nFocus: {focus}\nSections: {titles}\n\nDOCUMENTS:\n{docs}",
                              max_tokens=3000, model=self.fast_model)
            by = {x.id: [] for x in chunk}
            for it in d.get("items", []):
                sid = it.get("source") if it.get("source") in by else (chunk[0].id if len(chunk) == 1 else None)
                if sid:
                    by[sid].append(it)
            for x in chunk:
                out += self._add_items(by[x.id], x, titles)
        return out

    def extract_from(self, src: Source, focus: str = "") -> List[Fact]:
        return self.extract_many([src], focus)

    def extract_all(self):
        self.log("[6/9] Fact Extraction Agent")
        self.extract_many(list(self.s.sources.values()))
        self.log(f"      {len(self.s.facts)} structured facts/claims")

    # ---- Dynamic research expansion
    def expand(self):
        self.log("[+] Dynamic Research Expansion")
        s = self.s
        sample = "\n".join(f"- {f.text}" for f in s.facts[:40])
        d = self.llm.json(
            "expand",
            "Given a research question, the current section titles and extracted facts, identify up to 2 important "
            "SUBTOPICS that are mentioned in the evidence but not covered by any section. "
            "Output {\"new_sections\":[{\"title\":str,\"goal\":str,\"queries\":[2 queries]}]} (empty list if none).",
            f"Question: {s.query}\nSections: {[p['title'] for p in s.plan]}\nFacts:\n{sample}")
        for sec in d.get("new_sections", [])[:2]:
            if not sec.get("title") or not sec.get("queries"):
                continue
            self.log(f"      discovered subtopic: {sec['title']}")
            s.plan.append(sec)
            new = [x for x in self._search_for(sec["queries"], sec["title"], per_q=4) if x.id not in s.sources][:5]
            for x in new:
                s.sources[x.id] = x
            self.read_and_score(new, grow=True)
            self.extract_many([x for x in new if x.id in s.sources])

    # ---- Agent 7: cross-verification
    def _cred(self, fid_list):
        by = {f.id: f for f in self.s.facts}
        seen, tot = set(), 0.0
        for i in fid_list:
            f = by.get(i)
            if f and f.source_id not in seen and f.source_id in self.s.sources:
                seen.add(f.source_id)
                tot += self.s.sources[f.source_id].credibility
        return round(tot, 1), len(seen)

    def _listing(self, facts):
        return "\n".join(
            f"[{f.id}|src {f.source_id}|cred {self.s.sources[f.source_id].credibility:.0f}|{f.type}] {f.text}"
            for f in facts if f.source_id in self.s.sources)

    def verify(self):
        self.log("[7/9] Cross-Verification Agent")
        s = self.s
        facts = sorted([f for f in s.facts if f.source_id in s.sources],
                       key=lambda f: -s.sources[f.source_id].credibility)[:70]
        if len(facts) < 2:
            return
        d = self.llm.json(
            "verify",
            "You are a fact-checking analyst. Compare the numbered evidence items. Find (a) CONTRADICTIONS: items from "
            "different sources that disagree (different numbers, timelines, conclusions) and (b) AGREEMENTS supported by "
            "2+ sources. Output {\"contradictions\":[{\"topic\":str,\"claim_a\":{\"text\":str,\"fact_ids\":[ids]},"
            "\"claim_b\":{\"text\":str,\"fact_ids\":[ids]},\"why\":str}],\"agreements\":[{\"statement\":str,"
            "\"fact_ids\":[ids]}]}. Only use ids that exist. Only report genuine conflicts.",
            self._listing(facts), max_tokens=3000, effort="medium")
        valid = {f.id for f in s.facts}
        s.agreements = [a for a in d.get("agreements", []) if len([i for i in a.get("fact_ids", []) if i in valid]) >= 2][:10]
        cons = []
        for c in d.get("contradictions", []):
            a = [i for i in c.get("claim_a", {}).get("fact_ids", []) if i in valid]
            b = [i for i in c.get("claim_b", {}).get("fact_ids", []) if i in valid]
            if a and b and not set(a) & set(b):
                c["claim_a"]["fact_ids"], c["claim_b"]["fact_ids"] = a, b
                cons.append(c)
        self.log(f"      {len(cons)} contradiction(s) detected, {len(s.agreements)} agreement(s)")
        for c in cons[: self.max_contra]:
            self.resolve(c)
        s.contradictions = cons[: self.max_contra]

    def resolve(self, c):
        s = self.s
        self.log(f"      investigating: {c.get('topic')}")
        new_ids = []
        q = f"{c.get('topic', '')} {s.understanding.get('main_topic', '')}".strip()
        cand = [x for x in self.search_web(q, 5) + self.search_news(q, 2) if x.id not in s.sources][:4]
        for x in cand:
            x.section = s.plan[0]["title"]
            s.sources[x.id] = x
        self.read_and_score(cand, grow=True)
        new_ids += [f.id for f in self.extract_many([x for x in cand if x.id in s.sources], focus=c.get("topic", ""))]
        by = {f.id: f for f in s.facts}
        ev = [by[i] for i in c["claim_a"]["fact_ids"] + c["claim_b"]["fact_ids"] + new_ids if i in by]
        r = self.llm.json(
            "resolve",
            "You resolve conflicting research claims. Weigh source credibility (higher = more trustworthy), number of "
            "independent sources, recency and whether a claim is a forecast. Output {\"verdict\":\"A|B|partial|unresolved\","
            "\"resolution\":str (2-4 sentences, may give a consensus range),\"confidence\":0-100,"
            "\"new_evidence_ids\":[ids from the new evidence that support A or B]}.",
            f"Topic: {c.get('topic')}\nClaim A: {c['claim_a']['text']}\nClaim B: {c['claim_b']['text']}\n"
            f"Evidence (A ids {c['claim_a']['fact_ids']}, B ids {c['claim_b']['fact_ids']}, newly found {new_ids}):\n"
            f"{self._listing(ev)}", effort="medium")
        c["new_fact_ids"] = new_ids
        c["support_a"], c["n_sources_a"] = self._cred(c["claim_a"]["fact_ids"])
        c["support_b"], c["n_sources_b"] = self._cred(c["claim_b"]["fact_ids"])
        c["verdict"] = r.get("verdict", "unresolved")
        c["resolution"] = r.get("resolution", "")
        try:
            c["confidence"] = int(r.get("confidence", 50))
        except Exception:
            c["confidence"] = 50

    # ---- Agent 8: evidence graph
    def build_graph(self):
        self.log("[8/9] Knowledge Graph Agent")
        s, g = self.s, nx.DiGraph()
        root = s.understanding.get("main_topic") or s.query[:40]
        g.add_node("ROOT", label=root, kind="topic")
        for p in s.plan:
            g.add_node("SEC:" + p["title"], label=p["title"], kind="section")
            g.add_edge("ROOT", "SEC:" + p["title"], rel="covers")
        for src in s.sources.values():
            g.add_node(src.id, label=f"{domain_of(src.url)}\n({src.credibility:.0f})", kind="source",
                       credibility=src.credibility, url=src.url)
        for f in s.facts:
            if f.source_id not in s.sources:
                continue
            g.add_node(f.id, label=f.id, kind=f.type, text=f.text)
            g.add_edge("SEC:" + f.section, f.id, rel="evidence")
            g.add_edge(f.id, f.source_id, rel="from")
        for a in s.agreements:
            ids = [i for i in a["fact_ids"] if i in g]
            for x, y in zip(ids, ids[1:]):
                g.add_edge(x, y, rel="agrees")
        for c in s.contradictions:
            for x in c["claim_a"]["fact_ids"][:2]:
                for y in c["claim_b"]["fact_ids"][:2]:
                    if x in g and y in g:
                        g.add_edge(x, y, rel="contradicts")
        s.graph = g

    # ---- Agent 10: citation manager
    def ref_of(self, sid):
        s = self.s
        if sid not in s.refs:
            s.refs[sid] = len(s.refs) + 1
        return s.refs[sid]

    def _ev(self, facts):
        return "\n".join(
            f"- (id {f.id}, ref [{self.ref_of(f.source_id)}], reliability {self.s.sources[f.source_id].credibility:.0f}%, "
            f"{f.type}) {f.text}" for f in facts if f.source_id in self.s.sources)

    def references_md(self, text: str) -> str:
        s = self.s
        used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
        inv = {v: k for k, v in s.refs.items()}
        lines = []
        for n in used:
            src = s.sources.get(inv.get(n))
            if not src:
                continue
            lines.append(f"[{n}] {src.title or domain_of(src.url)} — {domain_of(src.url)}. "
                         f"{('Published: ' + str(src.published)[:10] + '. ') if src.published else ''}"
                         f"{src.url} — **Reliability: {src.credibility:.0f}%** ({src.cred_breakdown.get('category')})")
        return "\n\n".join(lines)

    # ---- Agent 9: report writer
    def write_report(self):
        self.log("[9/9] Report Writer + Citation Manager")
        s = self.s
        sys_w = ("You are a professional research analyst. Write ONLY from the evidence given; every sentence with a "
                 "factual claim must end with citation(s) like [3] or [2][5] using the ref numbers provided. "
                 "Flag forecasts as estimates, and mention when evidence is weak or from lower-reliability sources. "
                 "Do not invent facts. Write in clear markdown prose (short paragraphs, bullets allowed).")
        for p in s.plan:
            fs = [f for f in s.facts if f.section == p["title"]]
            if not fs:
                s.sections_md[p["title"]] = "_No sufficiently reliable evidence was found for this section._"
                continue
            s.sections_md[p["title"]] = self.llm.chat(
                sys_w, f"Section: {p['title']}\nGoal: {p['goal']}\nEvidence:\n{self._ev(fs)}\n\n"
                       f"Write the detailed findings for this section (150-300 words).", max_tokens=900)
        findings = "\n\n".join(f"### {t}\n{m}" for t, m in s.sections_md.items())
        top = sorted([f for f in s.facts if f.type == "fact" and f.source_id in s.sources],
                     key=lambda f: -s.sources[f.source_id].credibility)[:8]
        contra_md = self.contradictions_md()
        sep = "===SPLIT==="
        both = self.llm.chat(
            sys_w, f"Question: {s.query}\n\nFindings:\n{findings}\n\nKey evidence:\n{self._ev(top)}\n\n"
                   f"Write TWO parts separated by a line containing exactly {sep} (no 'PART' labels, no headings).\n"
                   f"First: a one-page Executive Summary (5-7 key-finding bullets, then a 2-sentence bottom line).\n"
                   f"Second: a short Background (80-150 words).", max_tokens=1300)
        exec_sum, _, background = both.partition(sep)
        if not background.strip():
            background = self.llm.chat(sys_w, f"Question: {s.query}\nEvidence:\n{self._ev(top)}\n\nWrite a short Background (80-150 words).", max_tokens=400)
        both2 = self.llm.chat(
            sys_w + " Analysis = reasoned interpretation, implications, scenarios, uncertainty; label your own inference "
                    "as 'Analysis' where it goes beyond sources.",
            f"Question: {s.query}\nFindings:\n{findings}\nContradictions:\n{contra_md}\n\n"
            f"Write TWO parts separated by a line containing exactly {sep} (no 'PART' labels).\n"
            f"First: the Analysis (200-350 words, with best/moderate/worst scenarios).\n"
            f"Second: the Conclusion (120-200 words: final assessment, confidence level, 2-3 recommended actions or open questions).",
            max_tokens=1800)
        analysis, _, concl = both2.partition(sep)
        if not concl.strip():
            concl = self.llm.chat(sys_w, f"Question: {s.query}\nAnalysis:\n{analysis}\n\nWrite the Conclusion (120-200 words).", max_tokens=500)
        body = (f"# Research Report: {s.query}\n\n*Generated {datetime.date.today()} · {len(s.sources)} sources · "
                f"{len(s.facts)} extracted facts · {len(s.contradictions)} contradictions investigated*\n\n"
                f"## Executive Summary\n{exec_sum}\n\n## Background\n{background}\n\n## Research Findings\n{findings}\n\n"
                f"## Contradictions & Conflicting Viewpoints\n{contra_md}\n\n## Analysis\n{analysis}\n\n"
                f"## Conclusion\n{concl}\n")
        s.report = body + "\n## References\n" + self.references_md(body)

    def contradictions_md(self):
        s = self.s
        if not s.contradictions:
            return "_No material contradictions were detected between reliable sources._"
        out = []
        for i, c in enumerate(s.contradictions, 1):
            rb = self._refs(c["claim_b"]["fact_ids"])
            out.append(f"**{i}. {c.get('topic')}**\n\n- **Claim A:** {c['claim_a']['text']} {self._refs(c['claim_a']['fact_ids'])}"
                       f" (credibility-weighted support: {c['support_a']}, {c['n_sources_a']} source(s))\n"
                       f"- **Claim B:** {c['claim_b']['text']} {rb}"
                       f" (credibility-weighted support: {c['support_b']}, {c['n_sources_b']} source(s))\n"
                       f"- **Why they differ:** {c.get('why', '')}\n"
                       f"- **Resolution ({c['verdict']}, confidence {c['confidence']}%):** {c['resolution']}")
        return "\n\n".join(out)

    def _refs(self, ids):
        by = {f.id: f for f in self.s.facts}
        seen = []
        for i in ids[:3]:
            f = by.get(i)
            if f and f.source_id in self.s.sources:
                r = self.ref_of(f.source_id)
                if r not in seen:
                    seen.append(r)
        return "".join(f"[{r}]" for r in seen)

    # ---- Interactive research mode
    def ask(self, question: str, session: Optional[Session] = None) -> str:
        """Answer follow-ups from stored evidence; do a mini-search if evidence is thin. No restart."""
        self.s = s = session or self.s
        qt = tokens(question)

        def rank():
            sc = [(len(qt & tokens(f.text + " " + f.section)), f) for f in s.facts if f.source_id in s.sources]
            return sorted(sc, key=lambda x: (-x[0], -s.sources[x[1].source_id].credibility))

        top = rank()
        if not top or top[0][0] < 2:
            self.log("      thin evidence -> running targeted mini-research")
            new = [x for x in self.search_web(f"{question} {s.understanding.get('main_topic', '')}", 5)
                   + self.search_arxiv(question, 2) if x.id not in s.sources][:4]
            for x in new:
                x.section = s.plan[0]["title"] if s.plan else ""
                s.sources[x.id] = x
            self.read_and_score(new, grow=True)
            self.extract_many([x for x in new if x.id in s.sources], focus=question)
            top = rank()
        ev = [f for _, f in top[:12]]
        ans = self.llm.chat(
            "Answer the follow-up question using ONLY the evidence; cite with [n]. If the evidence is insufficient, say so.",
            f"Original research question: {s.query}\nFollow-up: {question}\nEvidence:\n{self._ev(ev)}", max_tokens=900)
        refs = self.references_md(ans)
        return ans + ("\n\n**References**\n\n" + refs if refs else "")

    def expand_section(self, title: str, session: Optional[Session] = None) -> str:
        self.s = s = session or self.s
        match = next((p for p in s.plan if title.lower() in p["title"].lower()), None)
        if not match:
            return f"Section '{title}' not found. Sections: {[p['title'] for p in s.plan]}"
        fs = [f for f in s.facts if f.section == match["title"]]
        related = [f for f in s.facts if f.section != match["title"] and len(tokens(match["title"]) & tokens(f.text)) >= 2]
        out = self.llm.chat(
            "You are a research analyst. Explain this section in much greater depth (400-600 words) using ONLY the evidence; "
            "cite [n] after each claim; note uncertainties.",
            f"Section: {match['title']}\nGoal: {match['goal']}\nEvidence:\n{self._ev(fs + related[:6])}", max_tokens=1500)
        refs = self.references_md(out)
        return out + ("\n\n**References**\n\n" + refs if refs else "")


# ----------------------------------------------------------------------------- dashboards / viz
def contradiction_table(s: Session):
    import pandas as pd
    rows = []
    for c in s.contradictions:
        rows.append({"Topic": c.get("topic"), "Claim A": c["claim_a"]["text"], "Claim B": c["claim_b"]["text"],
                     "Support A (cred-weighted)": c["support_a"], "Support B (cred-weighted)": c["support_b"],
                     "Verdict": c["verdict"], "Confidence %": c["confidence"], "Resolution": c["resolution"]})
    return pd.DataFrame(rows)


def sources_table(s: Session):
    import pandas as pd
    rows = [{"Ref": s.refs.get(x.id, ""), "Title": x.title, "Domain": domain_of(x.url),
             "Reliability %": x.credibility, "Category": x.cred_breakdown.get("category"),
             "Peer reviewed": x.cred_breakdown.get("peer_reviewed"), "Published": x.published, "URL": x.url}
            for x in s.sources.values()]
    rows += [{"Ref": "✗", "Title": x.title, "Domain": domain_of(x.url), "Reliability %": x.credibility,
              "Category": "FILTERED OUT", "Peer reviewed": False, "Published": x.published, "URL": x.url}
             for x in s.rejected]
    return pd.DataFrame(rows).sort_values("Reliability %", ascending=False)


def draw_graph(s: Session, figsize=(13, 9)):
    import matplotlib
    import matplotlib.pyplot as plt
    g = s.graph
    fig, ax = plt.subplots(figsize=figsize)
    if g is None or len(g) == 0:
        return fig
    pos = nx.spring_layout(g, seed=7, k=0.55)
    colors = {"topic": "#1f3a8a", "section": "#2563eb", "source": "#16a34a", "fact": "#9ca3af",
              "statistic": "#f59e0b", "claim": "#a855f7"}
    sizes = {"topic": 1800, "section": 1100, "source": 700}
    nodes = list(g.nodes)
    nx.draw_networkx_nodes(g, pos, nodelist=nodes, ax=ax,
                           node_color=[colors.get(g.nodes[n]["kind"], "#999") for n in nodes],
                           node_size=[sizes.get(g.nodes[n]["kind"], 160) for n in nodes], alpha=0.9)
    for rel, col, st in [("covers", "#9ca3af", "solid"), ("evidence", "#d1d5db", "solid"),
                         ("from", "#86efac", "solid"), ("agrees", "#22c55e", "dashed"), ("contradicts", "#ef4444", "solid")]:
        es = [(u, v) for u, v, d in g.edges(data=True) if d["rel"] == rel]
        nx.draw_networkx_edges(g, pos, edgelist=es, ax=ax, edge_color=col, style=st, alpha=0.7,
                               width=2.2 if rel == "contradicts" else 0.8, arrows=rel != "agrees")
    nx.draw_networkx_labels(g, pos, {n: g.nodes[n]["label"] for n in nodes}, font_size=6, ax=ax)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=k)
                       for k, c in colors.items()] +
                      [Line2D([0], [0], color="#ef4444", lw=2, label="contradicts")], loc="lower left", fontsize=8)
    ax.set_title("Evidence Graph: topic → sections → facts → sources (red = contradiction)")
    ax.axis("off")
    return fig
