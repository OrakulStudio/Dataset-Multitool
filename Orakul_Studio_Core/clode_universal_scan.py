# ==============================================================================
#  ORAKUL CORE — AI Dataset & Model MultiTool
#  Module: Caption QC Scanner
# ------------------------------------------------------------------------------
#  Author:      Orakul (Orakul Studio)
#  GitHub:      https://github.com/OrakulStudio
#  Civitai:     https://civitai.com/user/ORAKUL_STUDIO
#  HuggingFace: https://huggingface.co/OrakulStorm
# ------------------------------------------------------------------------------
#  Description: Universal caption quality scanner with statistical analysis.
#  Copyright (c) 2026 Orakul Studio. All rights reserved.
# ==============================================================================

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from rich.console import Console
from rich.panel import Panel
from rich import box

def t(text: str) -> str:
    """Translation helper. Currently returns English. Ready for future i18n."""
    config_file = Path("config.json")
    lang = "en"
    
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                lang = json.load(f).get("language", "en")
        except Exception:
            pass
            
    if lang == "en" or not text:
        return text
        
    lang_file = Path(f"{lang}.json")
    if not lang_file.exists():
        return text
        
    try:
        with open(lang_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            data = {k.replace('\ufe0f', ''): v for k, v in raw_data.items()}
            
        clean_text = text.replace('\ufe0f', '')
            
        if clean_text in data:
            return data[clean_text]
            
        if "\n" in clean_text:
            lines = clean_text.split("\n")
            translated_lines = [data.get(line.replace('\ufe0f', ''), line) for line in lines]
            return "\n".join(translated_lines)
    except Exception:
        pass
        
    return text

console = Console()

HEADER_RE = re.compile(r'(?:^|[;.]\s*|,\s*)([a-z][a-z \-]{2,45}?):\s')
WORD_RE = re.compile(r"[a-zA-Z]+")

GENERIC_STOPWORDS = {
    "the", "and", "with", "from", "this", "that", "into", "their",
    "these", "those", "very", "only", "both", "such", "more", "some",
    "than", "then", "each", "where", "when", "what", "which", "while",
    "over", "under", "through", "between", "across", "within", "along",
    "near", "around", "also", "have", "will", "would", "could", "should",
}

def learn_headers(records: list[dict], min_files: int) -> set[str]:
    phrase_files = {}
    for r in records:
        lowered = r["text"].lower()
        found_in_file = set(m.group(1).strip() for m in HEADER_RE.finditer(lowered))
        for phrase in found_in_file:
            phrase_files.setdefault(phrase, set()).add(r["file"])
    return {p for p, files in phrase_files.items() if len(files) >= min_files}

def header_vocab_words(headers: set[str]) -> set[str]:
    words = set()
    for phrase in headers:
        words.update(WORD_RE.findall(phrase))
    return words

def check_headers(text: str, headers: set[str]) -> list[str]:
    lowered = text.lower()
    reasons = []
    bare_leaks = []
    for phrase in headers:
        total = lowered.count(phrase)
        as_header = lowered.count(phrase + ":")
        if total == 0:
            continue
        if total >= 2:
            reasons.append(f"REPEATED HEADER '{phrase}' ×{total}")
        if total > as_header:
            bare_leaks.append(phrase)
    if bare_leaks:
        reasons.append("HEADER LEAK WITHOUT COLON: " + ", ".join(bare_leaks))
    return reasons

def near_duplicate_halves(text: str, threshold: float = 0.6) -> bool:
    words = text.split()
    if len(words) < 20:
        return False
    mid = len(words) // 2
    a = set(w.lower() for w in words[:mid])
    b = set(w.lower() for w in words[mid:])
    if not a or not b:
        return False
    return len(a & b) / len(a | b) > threshold

def excessive_word_repeat(text: str, exclude: set[str], min_count: int = 12, max_ratio: float = 0.07) -> list[str]:
    words = [w.lower() for w in WORD_RE.findall(text)]
    if not words:
        return []
    counts = Counter(words)
    total = len(words)
    flagged = []
    for word, count in counts.items():
        if len(word) < 4 or word in exclude or word in GENERIC_STOPWORDS:
            continue
        if count >= min_count and (count / total) > max_ratio:
            flagged.append(f"{word}×{count}")
    return flagged

def scan_folder(folder: Path, min_header_files: int):
    files = sorted(folder.glob("*.txt"))
    if not files:
        console.print(Panel(
            f"[bold white]\\[ERROR][/bold white] [dim]No .txt files found in folder: {folder}[/dim]",
            border_style="grey50",
            box=box.ROUNDED
        ))
        sys.exit(1)

    records = [
        {"file": f.name, "text": f.read_text(encoding="utf-8", errors="ignore").strip()}
        for f in files
    ]
    for r in records:
        r["word_count"] = len(r["text"].split())

    headers = learn_headers(records, min_header_files)
    exclude_words = header_vocab_words(headers)

    wc = [r["word_count"] for r in records]
    avg, sd = mean(wc), (stdev(wc) if len(wc) > 1 else 0)

    flagged = []
    for r in records:
        reasons = check_headers(r["text"], headers)

        if near_duplicate_halves(r["text"]):
            reasons.append("TEXT LOOPING (first and second half match >60%)")

        repeats = excessive_word_repeat(r["text"], exclude_words)
        if repeats:
            reasons.append("WORD SPAM: " + ", ".join(repeats))

        if sd > 0:
            z = (r["word_count"] - avg) / sd
            if abs(z) > 2.2:
                direction = "long" if z > 0 else "short"
                reasons.append(f"ANOMALOUS LENGTH (too {direction}, {r['word_count']} words vs average {avg:.0f})")

        if reasons:
            flagged.append({
                "Filename": r["file"], 
                "Word_count": r["word_count"], 
                "Issues": " | ".join(reasons)
            })

    return flagged, len(records), headers

def main():
    ap = argparse.ArgumentParser(description="Universal caption quality scanner")
    ap.add_argument("folder", type=Path, nargs="?", default=Path("."), help="Path to folder with .txt files")
    ap.add_argument("--min-header-files", type=int, default=None, help="Minimum files for a header to be considered systemic")
    args = ap.parse_args()

    target_folder = args.folder.resolve()
    files_count = len(list(target_folder.glob("*.txt")))
    
    if files_count == 0:
        console.print(Panel(
            f"[bold white]\\[ERROR][/bold white] [dim]No .txt files found in folder: {target_folder}[/dim]",
            border_style="grey50",
            box=box.ROUNDED
        ))
        sys.exit(1)

    min_header_files = args.min_header_files or max(3, round(files_count * 0.10))

    with console.status("[dim]Scanning dataset and analyzing structure...[/dim]", spinner="dots"):
        flagged, total, headers = scan_folder(target_folder, min_header_files)

    out_path = target_folder / "caption_qc_report.csv"
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["Filename", "Word_count", "Issues"])
        writer.writeheader()
        writer.writerows(flagged)

    panel_text = (
        f"[white]Files scanned:[/white] [dim]{total}[/dim]\n"
        f"[white]System headers detected:[/white] [dim]{len(headers)}[/dim]\n"
        f"[white]Files with issues:[/white] [dim]{len(flagged)} ({(len(flagged)/total)*100:.1f}%)[/dim]\n"
        f"[white]Report saved:[/white] [dim]{out_path.name}[/dim]"
    )

    console.print(Panel(
        panel_text,
        title="[bold white] QC SCANNER RESULTS [/bold white]",
        border_style="grey50",
        box=box.ROUNDED
    ))

    if flagged:
        console.print("\n[bold white]Top found issues:[/bold white]")
        for r in flagged[:15]:
            console.print(f"  • [white]{r['Filename']}[/white]: [dim]{r['Issues']}[/dim]")
        if len(flagged) > 15:
            console.print(f"  [dim]... and {len(flagged) - 15} more files (see CSV report)[/dim]")

if __name__ == "__main__":
    main()