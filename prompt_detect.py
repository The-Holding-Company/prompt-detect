#!/usr/bin/env python3
"""prompt-detect — scan files in CWD for likely LLM prompt content.

Walks a directory, runs heuristics over text files, and reports each file's
likelihood of containing an LLM prompt (system prompt, instruction, few-shot
template, or jailbreak/injection payload).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".sh", ".bash",
    ".zsh", ".fish", ".rb", ".go", ".rs", ".java", ".kt", ".swift", ".c",
    ".h", ".cpp", ".hpp", ".cs", ".php", ".html", ".xml", ".jinja", ".j2",
    ".tmpl", ".tpl", ".prompt", ".prompty", ".mustache", ".hbs", ".handlebars",
}

SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache",
    ".pytest_cache", "dist", "build", ".next", ".turbo", ".cache", "target",
    ".idea", ".vscode",
}

MAX_BYTES = 512 * 1024  # 512 KiB per file

ROLE_PATTERNS = [
    re.compile(r"\byou\s+are\s+(an?\s+)?(helpful|expert|professional|principal|senior|world[- ]class)\b", re.I),
    re.compile(r"\byou\s+are\s+(claude|chatgpt|gpt-?\d|gemini|llama|mistral|an?\s+ai|an?\s+assistant|an?\s+agent)\b", re.I),
    re.compile(r"\byour\s+(role|task|job|goal|objective|purpose)\s+is\b", re.I),
    re.compile(r"\bact\s+as\b", re.I),
    re.compile(r"\bact\s+like\b", re.I),
    re.compile(r"\bpretend\s+(to\s+be|you\s+are)\b", re.I),
    re.compile(r"\brespond\s+as\b", re.I),
]

INSTRUCTION_PATTERNS = [
    re.compile(r"\b(must|never|always|do\s+not|don'?t)\s+(answer|respond|reveal|disclose|share|use|include|mention|generate|output|say|write|refuse)\b", re.I),
    re.compile(r"\bfollow\s+(these|the\s+following)\s+(rules|instructions|steps|guidelines)\b", re.I),
    re.compile(r"\bsystem\s+prompt\b", re.I),
    re.compile(r"\b(user|assistant|system)\s*:\s*", re.I),
    re.compile(r"<\|(system|user|assistant|im_start|im_end|begin_of_text|eot_id|start_header_id|end_header_id)\|>"),
    re.compile(r"<(system|user|assistant|instructions?|examples?|context|task|persona|role|prompt|input|output)>", re.I),
]

TEMPLATE_PATTERNS = [
    re.compile(r"\{\{\s*[a-zA-Z_][\w\.]*\s*\}\}"),                # jinja {{ var }}
    re.compile(r"\{%\s*(if|for|endif|endfor|block|extends)\b"),    # jinja control
    re.compile(r"\{[a-zA-Z_][\w]{0,40}\}"),                       # f-string / .format
    re.compile(r"\$\{[a-zA-Z_][\w]*\}"),                          # ${var}
]

FEWSHOT_PATTERNS = [
    re.compile(r"^\s*(Q|Question)\s*:", re.M),
    re.compile(r"^\s*(A|Answer)\s*:", re.M),
    re.compile(r"^\s*Example\s*\d*\s*:", re.I | re.M),
    re.compile(r"^\s*Input\s*:.*\n\s*Output\s*:", re.I | re.M),
]

INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above|preceding)\s+(instructions|prompts|rules)\b", re.I),
    re.compile(r"\bdisregard\s+(all\s+)?(previous|prior|above)\b", re.I),
    re.compile(r"\b(reveal|print|show|output)\s+(your|the)\s+(system\s+)?prompt\b", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\bDAN\b.*\bmode\b", re.I),
    re.compile(r"\bdeveloper\s+mode\b", re.I),
]

API_PATTERNS = [
    re.compile(r"\b(anthropic|openai|cohere|mistralai|google\.generativeai|langchain|llama_index|litellm)\b", re.I),
    re.compile(r"\b(messages|chat)\.create\b"),
    re.compile(r"\brole\s*[:=]\s*['\"](system|user|assistant)['\"]"),
    re.compile(r"\b(system_prompt|system_message|instructions|prompt_template)\b", re.I),
]

WEIGHTS = {
    "role":         3.0,
    "instruction":  2.0,
    "template":     0.8,
    "fewshot":      1.5,
    "injection":    2.5,
    "api":          1.0,
}


@dataclass
class Finding:
    path: str
    size: int
    score: float
    label: str
    signals: dict[str, int] = field(default_factory=dict)
    samples: list[str] = field(default_factory=list)


def looks_binary(blob: bytes) -> bool:
    if b"\x00" in blob[:4096]:
        return True
    # Heuristic: many non-text bytes in first 4 KiB
    sample = blob[:4096]
    if not sample:
        return False
    text_bytes = sum(1 for b in sample if 9 <= b <= 13 or 32 <= b < 127)
    return text_bytes / len(sample) < 0.85


def count_hits(text: str, patterns: list[re.Pattern]) -> tuple[int, list[str]]:
    hits = 0
    samples: list[str] = []
    for pat in patterns:
        for m in pat.finditer(text):
            hits += 1
            if len(samples) < 2:
                start = max(0, m.start() - 20)
                end = min(len(text), m.end() + 20)
                snippet = text[start:end].replace("\n", " ").strip()
                if len(snippet) > 120:
                    snippet = snippet[:117] + "..."
                samples.append(snippet)
            if hits >= 50:  # cap per pattern group
                return hits, samples
    return hits, samples


def score_text(text: str) -> tuple[float, dict[str, int], list[str]]:
    signals: dict[str, int] = {}
    samples: list[str] = []
    score = 0.0
    for name, patterns in (
        ("role", ROLE_PATTERNS),
        ("instruction", INSTRUCTION_PATTERNS),
        ("template", TEMPLATE_PATTERNS),
        ("fewshot", FEWSHOT_PATTERNS),
        ("injection", INJECTION_PATTERNS),
        ("api", API_PATTERNS),
    ):
        hits, snip = count_hits(text, patterns)
        if hits:
            signals[name] = hits
            score += WEIGHTS[name] * (1 + 0.3 * (hits - 1))  # diminishing returns
            samples.extend(snip[:1])
    return round(score, 2), signals, samples[:4]


def label_for(score: float, signals: dict[str, int]) -> str:
    if "injection" in signals and signals["injection"] >= 1:
        return "prompt-injection-suspect"
    if score >= 6:
        return "likely-prompt"
    if score >= 3:
        return "possible-prompt"
    if score >= 1:
        return "weak-signal"
    return "no-signal"


def iter_files(root: Path, all_files: bool):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            p = Path(dirpath) / name
            if not all_files:
                if p.suffix.lower() not in TEXT_EXTS:
                    continue
            yield p


def scan_file(path: Path) -> Finding | None:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size == 0:
        return None
    try:
        with path.open("rb") as f:
            blob = f.read(MAX_BYTES)
    except OSError:
        return None
    if looks_binary(blob):
        return None
    try:
        text = blob.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        return None
    score, signals, samples = score_text(text)
    if score == 0:
        return None
    return Finding(
        path=str(path),
        size=size,
        score=score,
        label=label_for(score, signals),
        signals=signals,
        samples=samples,
    )


def format_human(findings: list[Finding], threshold: float) -> str:
    if not findings:
        return "no prompt-like content detected"
    lines = []
    for f in findings:
        if f.score < threshold:
            continue
        sig = ", ".join(f"{k}={v}" for k, v in sorted(f.signals.items()))
        lines.append(f"{f.score:>6.2f}  {f.label:<24}  {f.path}")
        lines.append(f"        signals: {sig}")
        for s in f.samples:
            lines.append(f"        ~ {s}")
    return "\n".join(lines) if lines else f"no findings at score >= {threshold}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="prompt-detect",
        description="Detect whether files in CWD contain LLM prompt content.",
    )
    ap.add_argument("path", nargs="?", default=".", help="Root directory to scan (default: CWD)")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of human output")
    ap.add_argument("--threshold", type=float, default=1.0, help="Minimum score to report (default: 1.0)")
    ap.add_argument("--all-files", action="store_true", help="Scan all files, not just known text extensions")
    ap.add_argument("--top", type=int, default=0, help="Only show top-N findings by score (0 = all)")
    args = ap.parse_args(argv)

    root = Path(args.path).resolve()
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        return 2

    findings: list[Finding] = []
    if root.is_file():
        f = scan_file(root)
        if f:
            findings.append(f)
    else:
        for p in iter_files(root, args.all_files):
            f = scan_file(p)
            if f:
                findings.append(f)

    findings.sort(key=lambda f: f.score, reverse=True)
    findings = [f for f in findings if f.score >= args.threshold]
    if args.top:
        findings = findings[: args.top]

    if args.json:
        print(json.dumps([asdict(f) for f in findings], indent=2))
    else:
        print(format_human(findings, args.threshold))

    return 0 if findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
