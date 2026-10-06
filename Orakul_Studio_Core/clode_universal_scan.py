#!/usr/bin/env python3
"""
Универсальный сканер качества кэпшенов (Caption QC Scanner) с поддержкой rich.
"""

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

# --- ФУНКЦИЯ ПЕРЕВОДА ИЗ ГЛАВНОГО ФАЙЛА ---
def t(text: str) -> str:
    """Чистый переводчик: читает config.json, открывает {lang}.json и возвращает текст"""
    config_file = Path("config.json")
    lang = "ru"
    
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                lang = json.load(f).get("language", "ru")
        except Exception:
            pass
            
    if lang == "ru" or not text:
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
# -----------------------------------------

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
            reasons.append(t("ПОВТОР ЗАГОЛОВКА '") + phrase + t("' ×") + str(total))
        if total > as_header:
            bare_leaks.append(phrase)
    if bare_leaks:
        reasons.append(t("УТЕЧКА ЗАГОЛОВКА БЕЗ ДВОЕТОЧИЯ: ") + ", ".join(bare_leaks))
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
        console.print(t("[bold red][-] Файлы .txt не найдены в папке: ") + str(folder) + t("[/bold red]"))
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
            reasons.append(t("ЗАЦИКЛИВАНИЕ ТЕКСТА (первая и вторая половина совпадают >60%)"))

        repeats = excessive_word_repeat(r["text"], exclude_words)
        if repeats:
            reasons.append(t("СПАМ ОДНОГО СЛОВА: ") + ", ".join(repeats))

        if sd > 0:
            z = (r["word_count"] - avg) / sd
            if abs(z) > 2.2:
                direction = t("длинный") if z > 0 else t("короткий")
                reasons.append(t("АНОМАЛЬНАЯ ДЛИНА (слишком ") + direction + t(", ") + str(r['word_count']) + t(" слов при среднем ") + f"{avg:.0f}" + t(")"))

        if reasons:
            flagged.append({
                t("Имя_файла"): r["file"], 
                t("Количество_слов"): r["word_count"], 
                t("Причины_замечаний"): " | ".join(reasons)
            })

    return flagged, len(records), headers

def main():
    ap = argparse.ArgumentParser(description=t("Универсальный сканер качества кэпшенов"))
    ap.add_argument("folder", type=Path, nargs="?", default=Path("."), help=t("Путь к папке с .txt файлами"))
    ap.add_argument("--min-header-files", type=int, default=None, help=t("Минимум файлов для заголовка"))
    args = ap.parse_args()

    target_folder = args.folder.resolve()
    files_count = len(list(target_folder.glob("*.txt")))
    
    if files_count == 0:
        console.print(t("[bold red][-] В папке ") + str(target_folder) + t(" нет .txt файлов![/bold red]"))
        sys.exit(1)

    min_header_files = args.min_header_files or max(3, round(files_count * 0.10))

    with console.status(t("[bold yellow]Сканирование датасета и анализ структур...[/bold yellow]"), spinner="dots"):
        flagged, total, headers = scan_folder(target_folder, min_header_files)

    out_path = target_folder / t("отчет_проверки_капшенов.csv")
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[t("Имя_файла"), t("Количество_слов"), t("Причины_замечаний")])
        writer.writeheader()
        writer.writerows(flagged)

    panel_text = (
        t("[bold cyan]Просканировано файлов:[/bold cyan] ") + str(total) + "\n" +
        t("[bold cyan]Определено системных заголовков:[/bold cyan] ") + str(len(headers)) + "\n" +
        t("[bold yellow]Файлов с замечаниями:[/bold yellow] ") + str(len(flagged)) + f" [dim]({(len(flagged)/total)*100:.1f}%)[/dim]\n" +
        t("[bold green]Отчёт сохранён:[/bold green] ") + str(out_path.name)
    )

    console.print(Panel(
        panel_text,
        title=t("[bold green]📊 РЕЗУЛЬТАТЫ СКАНЕРА QC[/bold green]"),
        border_style="green"
    ))

    if flagged:
        console.print("\n" + t("[bold yellow]🔍 ТОП НАЙДЕННЫХ ПРОБЛЕМ:[/bold yellow]"))
        for r in flagged[:15]:
            console.print(f"  • [bold white]{r[t('Имя_файла')]}[/bold white]: [red]{r[t('Причины_замечаний')]}[/red]")
        if len(flagged) > 15:
            console.print(t("  [dim]... и ещё ") + str(len(flagged) - 15) + t(" файлов смотрите в CSV-отчёте.[/dim]"))

if __name__ == "__main__":
    main()