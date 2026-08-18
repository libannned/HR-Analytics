import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

INPUT_FILE = Path("tweets_export.csv")
OUTPUT_FILE = Path("themes_export.csv")
TOP_N = 10

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his", "i", "if", "in", "is", "it", "its", "itself", "me", "my", "of", "on", "or", "our", "ours", "she", "that", "the", "their", "theirs", "them", "they", "this", "to", "us", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will", "with", "you", "your", "yours", "about", "after", "again", "all", "also", "am", "any", "because", "before", "being", "between", "both", "but", "can", "did", "do", "does", "doing", "down", "during", "each", "few", "further", "here", "how", "into", "more", "most", "other", "over", "same", "should", "some", "such", "than", "then", "there", "these", "those", "through", "too", "under", "up", "very", "via", "would", "could", "today", "tomorrow", "yesterday", "rt", "http", "https", "co", "com", "www", "amp", "x", "t", "s", "not", "re", "just", "new", "people", "so", "like", "one", "no", "now", "team", "great", "good", "think", "going", "get", "every", "many", "saying", "concerning", "interesting", "out", "go", "big", "week", "time", "yes", "don", "ve", "charts", "excited", "thank", "conversation", "see", "know", "need", "make", "way", "still", "even", "well", "back", "first", "last", "next", "long", "never", "always", "ever", "really", "actually", "already", "maybe", "probably", "literally", "basically", "something", "nothing", "everything", "anything", "someone", "anyone", "everyone", "thing", "things", "lot", "lots", "bit", "much", "little", "small", "right", "left", "high", "low", "old", "young", "best", "worst", "better", "worse", "different", "another", "others", "almost", "quite", "rather", "pretty", "enough", "around", "away", "off", "once", "twice", "together", "apart", "forward", "done", "got", "came", "went", "said", "made", "took", "put", "let", "run", "try", "ask", "seem", "feel", "keep", "start", "stop", "show", "give", "tell", "call", "play", "move", "live", "hold", "turn", "bring", "talk", "look", "come", "want", "use", "find", "work", "help", "say", "set", "own", "hey", "please", "welcome", "twitter", "realized", "year", "years", "day", "days", "weeks", "month", "months", "hour", "hours", "minute", "minutes", "second", "seconds", "congrats", "launch", "absolutely", "wow", "lol", "omg", "tbh", "imo", "fyi", "btw", "aka", "etc", "vs", "per", "ie", "eg", "nb", "glad", "hope", "wish", "love", "hate", "thought", "believe", "guess", "suppose", "wonder", "mean", "matter", "point", "fact", "case", "place", "part", "side", "kind", "sort", "type", "form", "level", "number", "amount", "percent", "million", "billion", "trillion", "thousand", "hundred"
}

THEME_BUCKETS = {
    "AI/ML": {"ai", "artificial", "intelligence", "llm", "llms", "model", "models", "ml", "machine", "learning", "agent", "agents", "gpt", "claude", "gemini", "openai", "anthropic", "mistral", "deepmind", "autonomous", "agentic", "inference", "training", "fine", "tuning", "embedding", "embeddings", "transformer", "transformers", "neural", "network", "networks", "deep", "generative", "multimodal", "reasoning", "rag", "vector", "database", "gpu", "compute", "robotics", "robot", "robots", "automation", "automated", "autopilot", "copilot", "assistant", "assistants", "chatbot", "chatbots", "prompt", "prompting", "context", "window", "tokens", "token", "synthetic", "foundation", "open", "source", "weights", "parameters", "benchmark", "evals", "evaluation", "alignment", "safety", "hallucination", "agi", "coding", "code", "programmer", "programming", "developer", "developers"},
    "Startups": {"startup", "startups", "founder", "founders", "company", "companies", "product", "build", "building", "early", "stage", "pre", "seed", "entrepreneur", "entrepreneurs", "venture", "backed", "portfolio", "operator", "operators", "hire", "hiring", "talent", "culture", "mission", "vision", "pitch", "deck", "demo", "mvp", "traction", "growth", "scale", "scaling", "pivot", "iterate", "feedback", "users", "customers", "retention", "churn", "acquisition", "b2b", "b2c", "saas", "marketplace", "platform", "network", "effects", "moat", "defensible", "category", "winner", "breakout", "unicorn", "decacorn", "exit", "ipo", "spac", "merger"},
    "Funding/Capital": {"funding", "fund", "capital", "round", "valuation", "invest", "investing", "investor", "investors", "seed", "series", "raise", "raising", "check", "checks", "term", "sheet", "deal", "deals", "close", "closed", "led", "participated", "syndicate", "angel", "angels", "money", "dollars", "equity", "dilution", "cap", "table", "priced", "safe", "note", "convertible", "bridge", "extension", "secondary", "liquidity", "carry", "returns", "irr", "moic", "multiple", "lp", "lps", "gp", "management", "fee", "allocation", "thesis", "mandate", "sector", "focus", "vertical", "horizontal"},
    "Engineering/Product": {"engineer", "engineering", "software", "platform", "infra", "infrastructure", "roadmap", "api", "cloud", "aws", "azure", "gcp", "devops", "mlops", "backend", "frontend", "fullstack", "mobile", "web", "app", "application", "system", "systems", "architecture", "design", "ux", "ui", "interface", "experience", "stack", "storage", "security", "privacy", "github", "deployment", "production", "testing", "debugging", "performance", "latency", "throughput", "reliability", "scalability", "tooling", "tools", "workflow", "pipeline", "integration", "plugin", "extension", "sdk", "framework", "library", "package"},
    "Markets/Macro": {"market", "markets", "economy", "economic", "macro", "inflation", "rates", "growth", "recession", "gdp", "fed", "interest", "yield", "curve", "cycle", "bull", "bear", "correction", "crash", "bubble", "boom", "bust", "trade", "tariff", "tariffs", "supply", "chain", "demand", "consumer", "enterprise", "smb", "government", "public", "private", "sector", "industry", "global", "domestic", "international", "emerging", "developed", "unemployment", "jobs", "labor", "workforce", "productivity", "efficiency", "output"},
    "Healthcare": {"health", "healthcare", "medical", "medicine", "clinical", "hospital", "patient", "patients", "doctor", "doctors", "physician", "nurses", "pharma", "pharmaceutical", "drug", "drugs", "biotech", "biology", "genomics", "diagnostic", "therapeutics", "treatment", "care", "insurance", "medicaid", "medicare", "reimbursement", "claims", "ehr", "emr", "records", "telemedicine", "telehealth", "mental", "behavioral", "wellness", "fitness", "longevity", "aging", "elderly"},
    "Crypto/Web3": {"crypto", "bitcoin", "ethereum", "web3", "blockchain", "token", "tokens", "defi", "nft", "nfts", "dao", "protocol", "wallet", "exchange", "dex", "stablecoin", "stablecoins", "layer", "rollup", "consensus", "validator", "mining", "staking", "liquidity", "pool", "smart", "contract", "contracts", "solidity", "rust", "solana", "polygon", "base", "arbitrum"},
    "Defense/Deeptech": {"defense", "military", "national", "security", "drone", "drones", "weapons", "surveillance", "satellite", "space", "rocket", "nuclear", "energy", "climate", "carbon", "renewable", "solar", "wind", "battery", "batteries", "ev", "electric", "vehicle", "vehicles", "cyber", "cybersecurity", "hacking", "vulnerability", "quantum", "computing", "photonics", "semiconductor", "chip", "chips"},
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[@#]", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    tokens = [tok for tok in normalize_text(text).split() if tok and tok not in STOPWORDS]
    # Keep tokens with at least 2 chars, but allow common abbreviations like ai.
    return [tok for tok in tokens if len(tok) > 1]


def extract_bigrams(tokens):
    bigrams = []
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if a in STOPWORDS or b in STOPWORDS:
            continue
        bigrams.append(f"{a} {b}")
    return bigrams


def assign_theme(keyword: str) -> str:
    words = set(keyword.split())
    for theme, vocab in THEME_BUCKETS.items():
        if words & vocab:
            return theme
    return "General"


def score_keyword(count: int, total_docs: int, doc_freq: int) -> float:
    # TF-IDF-like weighting to reduce generic terms, stable for small corpora.
    idf = math.log((1 + total_docs) / (1 + doc_freq)) + 1
    return count * idf


def read_rows(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"handle", "tweet text", "date", "likes", "retweets"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            rows.append(row)
    return rows


def build_handle_counters(rows):
    by_handle = defaultdict(list)
    for row in rows:
        by_handle[row["handle"].strip()].append(row)

    handle_keyword_counts = {}
    handle_docfreq = {}

    for handle, handle_rows in by_handle.items():
        unigram_counts = Counter()
        bigram_counts = Counter()
        doc_freq = Counter()

        for row in handle_rows:
            tokens = tokenize(row["tweet text"])
            bigrams = extract_bigrams(tokens)

            unigram_counts.update(tokens)
            bigram_counts.update(bigrams)

            seen_terms = set(tokens) | set(bigrams)
            doc_freq.update(seen_terms)

        merged = Counter()
        merged.update(unigram_counts)
        # Give bigrams slightly higher importance to capture "themes" better.
        for term, cnt in bigram_counts.items():
            merged[term] += int(cnt * 1.3)

        handle_keyword_counts[handle] = merged
        handle_docfreq[handle] = doc_freq

    return by_handle, handle_keyword_counts, handle_docfreq


def top_keywords_per_handle(by_handle, handle_keyword_counts, handle_docfreq, top_n=10):
    result = {}

    for handle, counts in handle_keyword_counts.items():
        total_docs = max(1, len(by_handle[handle]))
        docfreq = handle_docfreq[handle]

        scored = []
        for term, cnt in counts.items():
            df = docfreq.get(term, 1)
            scored.append((term, cnt, score_keyword(cnt, total_docs, df)))

        scored.sort(key=lambda x: (x[2], x[1], x[0]), reverse=True)
        result[handle] = scored[:top_n]

    return result


def find_cross_handle_signals(per_handle_top):
    term_to_handles = defaultdict(set)
    term_strength = defaultdict(float)

    for handle, items in per_handle_top.items():
        for term, raw_count, weighted_score in items:
            term_to_handles[term].add(handle)
            term_strength[term] += weighted_score + raw_count

    shared = []
    for term, handles in term_to_handles.items():
        if len(handles) >= 2:
            shared.append((term, len(handles), sorted(handles), term_strength[term]))

    shared.sort(key=lambda x: (x[1], x[3], x[0]), reverse=True)
    return shared


def write_output_csv(path: Path, per_handle_top, cross_signals):
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "section",
            "handle",
            "keyword_or_theme",
            "theme_bucket",
            "count_or_handles",
            "score",
            "handles",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for handle, items in per_handle_top.items():
            for term, raw_count, weighted_score in items:
                writer.writerow(
                    {
                        "section": "per_handle_top10",
                        "handle": handle,
                        "keyword_or_theme": term,
                        "theme_bucket": assign_theme(term),
                        "count_or_handles": raw_count,
                        "score": f"{weighted_score:.4f}",
                        "handles": "",
                    }
                )

        for term, handle_count, handles, strength in cross_signals:
            writer.writerow(
                {
                    "section": "cross_handle_signal",
                    "handle": "",
                    "keyword_or_theme": term,
                    "theme_bucket": assign_theme(term),
                    "count_or_handles": handle_count,
                    "score": f"{strength:.4f}",
                    "handles": "|".join(handles),
                }
            )


def print_summary(per_handle_top, cross_signals):
    print("\n=== Top 10 Themes/Keywords Per Handle ===")
    for handle in sorted(per_handle_top):
        print(f"\n{handle}")
        for idx, (term, raw_count, weighted_score) in enumerate(per_handle_top[handle], start=1):
            bucket = assign_theme(term)
            print(f"  {idx:>2}. {term:<28} count={raw_count:<4} score={weighted_score:>7.2f} theme={bucket}")

    print("\n=== Strongest Cross-Handle Signals (appearing in multiple handles) ===")
    if not cross_signals:
        print("No overlapping themes/keywords found across handles.")
        return

    for idx, (term, handle_count, handles, strength) in enumerate(cross_signals[:20], start=1):
        bucket = assign_theme(term)
        joined = ", ".join(handles)
        print(
            f"  {idx:>2}. {term:<28} handles={handle_count:<2} strength={strength:>8.2f} "
            f"theme={bucket} -> [{joined}]"
        )


def main():
    rows = read_rows(INPUT_FILE)
    by_handle, handle_keyword_counts, handle_docfreq = build_handle_counters(rows)
    per_handle_top = top_keywords_per_handle(by_handle, handle_keyword_counts, handle_docfreq, top_n=TOP_N)
    cross_signals = find_cross_handle_signals(per_handle_top)

    write_output_csv(OUTPUT_FILE, per_handle_top, cross_signals)
    print_summary(per_handle_top, cross_signals)

    print(f"\nWrote output to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
