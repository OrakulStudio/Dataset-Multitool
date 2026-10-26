# ==============================================================================
#  ORAKUL CORE — AI Dataset & Model MultiTool
#  Module: Auto Clean Dataset
# ------------------------------------------------------------------------------
#  Author:      Orakul (Orakul Studio)
#  GitHub:      https://github.com/OrakulStudio
#  Civitai:     https://civitai.com/user/ORAKUL_STUDIO
#  HuggingFace: https://huggingface.co/OrakulStorm
# ------------------------------------------------------------------------------
#  Description: Automatic caption cleaner based on QC Scanner report.
#  Copyright (c) 2026 Orakul Studio. All rights reserved.
# ==============================================================================

import csv
import json
import re
import sys
from pathlib import Path
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

def clean_duplicated_loop(text: str) -> str:
    words = text.split()
    if len(words) < 20:
        return text
    mid = len(words) // 2
    truncated = " ".join(words[:mid]).strip()
    last_punct = max(truncated.rfind('.'), truncated.rfind(';'))
    if last_punct > 50:
        return truncated[:last_punct].strip()
    return truncated

def clean_bare_leaks(text: str, reasons: str) -> str:
    match = re.search(r'(?:HEADER LEAK WITHOUT COLON|УТЕЧКА ЗАГОЛОВКА БЕЗ ДВОЕТОЧИЯ):\s*([^|]+)', reasons)
    if not match:
        return text
    phrases_str = match.group(1).strip()
    phrases = [p.strip() for p in phrases_str.split(',') if p.strip()]
    for phrase in phrases:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        text = pattern.sub('', text)
    return text

def clean_word_spam(text: str, reasons: str) -> str:
    text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)
    matches = re.findall(r'(\w+)[\×x]\d+', reasons)
    for word in matches:
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        occurrences = list(pattern.finditer(text))
        if len(occurrences) > 2:
            keep_until = occurrences[1].end()
            first_part = text[:keep_until]
            second_part = text[keep_until:]
            second_part_cleaned = pattern.sub('', second_part)
            text = first_part + second_part_cleaned
    return text

def clean_length_outliers(text: str, max_words: int = 140) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    
    truncated_words = words[:max_words]
    truncated_text = " ".join(truncated_words)
    
    last_punct = max(truncated_text.rfind('.'), truncated_text.rfind(';'))
    if last_punct > 50:
        return truncated_text[:last_punct + 1].strip()
    
    last_comma = truncated_text.rfind(',')
    if last_comma > 50:
        return truncated_text[:last_comma].strip()
        
    return truncated_text.strip()

def normalize_formatting(text: str) -> str:
    text = re.sub(r'\s*,\s*,\s*', ', ', text)
    text = re.sub(r'\s*\.\s*\.\s*', '. ', text)
    text = re.sub(r'^\s*[\s,.]+', '', text)
    text = re.sub(r'[\s,.]+$', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    try:
        target_folder = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
        
        csv_files = list(target_folder.glob("*.csv"))
        if not csv_files:
            console.print(Panel(
                f"[bold white]\\[ERROR][/bold white] [dim]CSV report not found in folder: {target_folder}[/dim]",
                border_style="grey50",
                box=box.ROUNDED
            ))
            return
            
        csv_path = csv_files[0]
        console.print(Panel(
            f"[bold white]\\[SYSTEM][/bold white] [dim]Found report: {csv_path.name}[/dim]",
            border_style="grey50",
            box=box.ROUNDED
        ))
        
        fixed_count = 0
        actions_log = []
        
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        if not rows:
            console.print(Panel(
                "[dim]CSV report is empty.[/dim]",
                border_style="grey50",
                box=box.ROUNDED
            ))
            return

        for row in rows:
            # Support both old Russian and new English CSV headers
            filename = (
                row.get("Filename") or 
                row.get("Имя_файла") or 
                row.get("File_name") or 
                row.get(t("Имя_файла"))
            )
            reasons = (
                row.get("Issues") or 
                row.get("Причины_замечаний") or 
                row.get("Reasons_for_remarks") or 
                row.get(t("Причины_замечаний")) or 
                ""
            )
            
            if not filename:
                continue
                
            file_path = target_folder / filename
            if not file_path.exists():
                actions_log.append(f"[dim]• {filename} → File not found on disk[/dim]")
                continue
                
            original_text = file_path.read_text(encoding="utf-8", errors="ignore")
            cleaned_text = original_text
            applied_fixes = []
            
            if "TEXT LOOPING" in reasons or "ЗАЦИКЛИВАНИЕ" in reasons:
                cleaned_text = clean_duplicated_loop(cleaned_text)
                applied_fixes.append("Smart loop trim")
            if "HEADER LEAK" in reasons or "УТЕЧКА" in reasons:
                cleaned_text = clean_bare_leaks(cleaned_text, reasons)
                applied_fixes.append("Removed header leaks")
            if "WORD SPAM" in reasons or "СПАМ" in reasons:
                cleaned_text = clean_word_spam(cleaned_text, reasons)
                applied_fixes.append("Removed word spam")
            
            if "ANOMALOUS LENGTH" in reasons or "АНОМАЛЬНАЯ ДЛИНА" in reasons:
                if "long" in reasons.lower() or "длинный" in reasons.lower():
                    cleaned_text = clean_length_outliers(cleaned_text, max_words=140)
                    applied_fixes.append("Trimmed excessive length")
                elif "short" in reasons.lower() or "короткий" in reasons.lower():
                    applied_fixes.append("Skipped: too short (needs manual fix)")
                else:
                    applied_fixes.append("Length anomaly")
                
            cleaned_text = normalize_formatting(cleaned_text)
            
            if cleaned_text != original_text:
                file_path.write_text(cleaned_text, encoding="utf-8")
                fixed_count += 1
                actions_log.append(f"[white]• {filename}[/white] → [dim]{' | '.join(applied_fixes)}[/dim]")
            else:
                if applied_fixes:
                    actions_log.append(f"[white]• {filename}[/white] → [dim]{' | '.join(applied_fixes)}[/dim]")
                else:
                    actions_log.append(f"[dim]• {filename} → No changes[/dim]")

        if actions_log:
            console.print(Panel(
                "\n".join(actions_log),
                title=f"[bold white] CLEANING REPORT (Files changed: {fixed_count}) [/bold white]",
                border_style="grey50",
                box=box.ROUNDED,
                expand=False
            ))
        else:
            console.print(Panel(
                "[dim]No files required changes.[/dim]",
                border_style="grey50",
                box=box.ROUNDED
            ))

    except Exception as e:
        console.print(Panel(
            f"[bold white]\\[ERROR][/bold white] [dim]Cleaning script error: {e}[/dim]",
            border_style="grey50",
            box=box.ROUNDED
        ))

if __name__ == "__main__":
    main()