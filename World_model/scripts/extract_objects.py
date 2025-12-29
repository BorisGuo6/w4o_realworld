import argparse
import json
import os
import re
from typing import List


def strip_prefix(text: str) -> str:
    # Remove leading "revise the image" (case-insensitive), plus trailing punctuation/space
    t = text.strip()
    m = re.match(r"^\s*revise\s+the\s+image[\s,:;-]*", t, flags=re.IGNORECASE)
    if m:
        t = t[m.end():].lstrip()
    return t


def extract_with_spacy(text: str) -> List[str]:
    try:
        import spacy  # type: ignore
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            # Model not installed
            return []
    except Exception:
        return []

    doc = nlp(text)

    # Collect candidates from noun chunks and noun/proper noun tokens
    candidates: List[str] = []
    seen = set()

    def push(token_text: str):
        key = token_text.strip().lower()
        if not key:
            return
        if key in {"the", "a", "an"}:
            return
        if key not in seen:
            seen.add(key)
            candidates.append(key)

    for chunk in doc.noun_chunks:
        # Clean determiners/articles on the left
        chunk_text = re.sub(r"^(the|a|an)\s+", "", chunk.text.strip(), flags=re.IGNORECASE)
        chunk_text = re.sub(r"[^a-zA-Z0-9_\-\s]", "", chunk_text).strip()
        if chunk_text:
            push(chunk_text)

    for tok in doc:
        if tok.pos_ in {"NOUN", "PROPN"} and not tok.is_stop:
            token_text = re.sub(r"[^a-zA-Z0-9_\-]", "", tok.text)
            if token_text:
                push(token_text)

    return candidates


def extract_moved_with_rules(text: str) -> List[str]:
    """Heuristic extraction: return only moved objects (direct objects of motion verbs)."""
    t = " ".join(text.lower().split())
    # Common motion verbs
    verb_pat = r"move|transfer|relocate|drag|shift|carry|bring|lift|put|place"
    # Patterns like: move the tomato to pan; put a cup on the table; carry apple into basket
    # Capture phrase after verb up to preposition
    m = re.search(rf"\b(?:{verb_pat})\b\s+(?:the\s+|a\s+|an\s+)?([^.,;:!?]+?)\s+\b(to|into|onto|on|in|inside|within|onto the|to the|into the|on the|in the)\b", t)
    candidates: List[str] = []
    if m:
        phrase = m.group(1).strip()
        # Keep last token as head noun; fallback to full phrase
        tokens = re.findall(r"[a-zA-Z0-9_\-]+", phrase)
        if tokens:
            candidates.append(tokens[-1])
        else:
            candidates.append(phrase)
    else:
        # Simpler pattern: verb + object (no explicit preposition)
        m2 = re.search(rf"\b(?:{verb_pat})\b\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z0-9_\-]+)", t)
        if m2:
            candidates.append(m2.group(1))

    # Deduplicate while preserving order
    seen = set()
    res: List[str] = []
    for c in candidates:
        k = c.strip().lower()
        if k and k not in seen:
            seen.add(k)
            res.append(k)
    return res


def extract_with_gemini(text: str) -> List[str]:
    """Use Gemini to extract ONLY the moved object(s) as a JSON array of strings."""

    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore


    api_key = os.environ.get("GEMINI_API_KEY") or "AIzaSyAIPxQUTVPKHG1A6U9QCviPiMvF8wz4lHY"
    if not api_key:
        return []
    try:
        client = genai.Client(api_key=api_key)
    except Exception:
        return []

    system_prompt = (
        "You are an information extraction assistant. Given an English instruction, "
        "identify ONLY the objects that are being moved/relocated by the user. "
        "Return a compact JSON array of lowercase strings (object nouns). "
        "Do not include destination/containers or locations. "
        "Examples:\n"
        "- 'revise the image, move the tomato to pan' -> [\"tomato\"]\n"
        "- 'put the apple into the bowl' -> [\"apple\"]\n"
        "- 'transfer three cups onto the tray' -> [\"cups\"]\n"
        "If no moved object can be identified, return []."
    )

    content = [
        genai_types.Content(role="user", parts=[
            genai_types.Part(text=system_prompt + "\nInstruction: " + text)
        ])
    ]
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=content,
            config=genai_types.GenerateContentConfig(response_mime_type="application/json")
        )
        raw = getattr(resp, "text", "") or ""
        raw = raw.strip()
        # Some SDKs may put JSON in candidates parts; try to parse resp.text
        return json.loads(raw)
    except Exception:
        return []


def extract_objects(text: str) -> List[str]:
    t = strip_prefix(text)
    # 1) Try Gemini for moved objects only
    res = extract_with_gemini(t)
    if res:
        return res
    # 2) Fallback: rule-based moved object extraction
    res = extract_moved_with_rules(t)
    if res:
        return res
    # 3) Last resort: spaCy nouns (then filter with a weak heuristic around motion verbs)
    nouns = extract_with_spacy(t)
    if nouns:
        return nouns[:1]  # pick the first as best guess for moved object
    return []


def main():
    parser = argparse.ArgumentParser(description="Extract object names from an instruction sentence.")
    parser.add_argument("instruction", type=str, help="input sentence, e.g., 'revise the image, move the tomato to pan'")
    parser.add_argument("--json", action="store_true", help="print JSON array instead of comma-separated text")
    args = parser.parse_args()

    objs = extract_objects(args.instruction)
    if args.json:
        print(json.dumps(objs, ensure_ascii=False))
    else:
        print(", ".join(objs))


if __name__ == "__main__":
    main()


