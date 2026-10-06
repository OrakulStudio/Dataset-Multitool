#!/usr/bin/env python3
"""
Скрипт автоматической очистки кэпшенов с подробным логированием каждого файла из отчёта.
"""

import csv
import json
import re
import sys
from pathlib import Path
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
    match = re.search(r'(?:УТЕЧКА ЗАГОЛОВКА БЕЗ ДВОЕТОЧИЯ|HEADER LEAK WITHOUT COLON):\s*([^|]+)', reasons)
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
            console.print(t("[bold red][-] CSV-отчёт не найден в папке: ") + str(target_folder) + t("[/bold red]"))
            return
            
        csv_path = csv_files[0]
        console.print(t("[bold cyan][*] Найден отчёт:[/bold cyan] ") + csv_path.name)
        
        fixed_count = 0
        actions_log = []
        
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        if not rows:
            console.print(t("[bold yellow][i] CSV-отчёт пуст.[/bold yellow]"))
            return

        for row in rows:
            # Читаем ключи с учетом возможного перевода CSV заголовков
            filename = row.get(t("Имя_файла")) or row.get("Имя_файла") or row.get("File_name")
            reasons = row.get(t("Причины_замечаний")) or row.get("Причины_замечаний") or row.get("Reasons_for_remarks") or ""
            
            if not filename:
                continue
                
            file_path = target_folder / filename
            if not file_path.exists():
                actions_log.append(t("[bold red]• ") + filename + t(" ➔ Файл не найден на диске[/bold red]"))
                continue
                
            original_text = file_path.read_text(encoding="utf-8", errors="ignore")
            cleaned_text = original_text
            applied_fixes = []
            
            if "ЗАЦИКЛИВАНИЕ" in reasons or "TEXT LOOPING" in reasons:
                cleaned_text = clean_duplicated_loop(cleaned_text)
                applied_fixes.append(t("Умная обрезка цикла"))
            if "УТЕЧКА" in reasons or "HEADER LEAK" in reasons:
                cleaned_text = clean_bare_leaks(cleaned_text, reasons)
                applied_fixes.append(t("Удалены утечки заголовков"))
            if "СПАМ" in reasons or "SPAM" in reasons:
                cleaned_text = clean_word_spam(cleaned_text, reasons)
                applied_fixes.append(t("Убран спам слов"))
            
            if "АНОМАЛЬНАЯ ДЛИНА" in reasons or "ANOMALOUS LENGTH" in reasons:
                if t("длинный") in reasons.lower() or "long" in reasons.lower():
                    cleaned_text = clean_length_outliers(cleaned_text, max_words=140)
                    applied_fixes.append(t("Обрезан избыточный объем"))
                elif t("короткий") in reasons.lower() or "short" in reasons.lower():
                    applied_fixes.append(t("[yellow]Пропущен: слишком короткий (требует ручного долива)[/yellow]"))
                else:
                    applied_fixes.append(t("Аномалия длины"))
                
            cleaned_text = normalize_formatting(cleaned_text)
            
            if cleaned_text != original_text:
                file_path.write_text(cleaned_text, encoding="utf-8")
                fixed_count += 1
                actions_log.append(t("[bold white]• ") + filename + t("[/bold white] ➔ [green]") + " | ".join(applied_fixes) + t("[/green]"))
            else:
                if applied_fixes:
                    actions_log.append(t("[bold white]• ") + filename + t("[/bold white] ➔ [yellow]") + " | ".join(applied_fixes) + t("[/yellow]"))
                else:
                    actions_log.append(t("[bold white]• ") + filename + t("[/bold white] ➔ [dim]Без изменений[/dim]"))

        if actions_log:
            console.print(Panel(
                "\n".join(actions_log),
                title=t("[bold green]✨ ИТОГОВЫЙ ОТЧЁТ ОЧИСТКИ (Изменено файлов: ") + str(fixed_count) + t(")[/bold green]"),
                border_style="green",
                expand=False
            ))
        else:
            console.print(t("\n[bold yellow][i] Ни один файл не потребовал изменений.[/bold yellow]"))

    except Exception as e:
        console.print(t("[bold red][-] Ошибка в скрипте очистки: ") + str(e) + t("[/bold red]"))

if __name__ == "__main__":
    main()