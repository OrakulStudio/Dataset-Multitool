# ==============================================================================
#  ORAKUL CORE — AI Dataset & Model MultiTool
#  Module: Model Inspector & Metadata Analyzer
# ------------------------------------------------------------------------------
#  Author:      Orakul (Orakul Studio)
#  GitHub:      https://github.com/OrakulStudio
#  Civitai:     https://civitai.com/user/ORAKUL_STUDIO
#  HuggingFace: https://huggingface.co/OrakulStorm
# ------------------------------------------------------------------------------
#  Description: Fast safetensors header reading, rank & parameter counter.
#  Copyright (c) 2026 Orakul Studio. All rights reserved.
# ==============================================================================
import os
import math
import json
import time
from pathlib import Path
from safetensors import safe_open
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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

def clean_path(path_str: str) -> str:
    """Очистка пути при Drag-and-Drop из проводника"""
    return path_str.strip(' "\'')

def inspect_model():
    console.print(Panel(t("[bold yellow]🔍 ORAKUL MODEL INSPECTOR • PARAMS & METADATA 🔍[/bold yellow]"), expand=False))
    
    # 1. Запрос пути к файлу
    raw_path = console.input("\n" + t("[bold cyan]Перетащи модель (.safetensors) сюда: [/bold cyan]"))
    file_path = clean_path(raw_path)

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        console.print(t("[bold red]❌ Ошибка: Файл не найден![/bold red]"))
        return

    start_time = time.time()
    total_params = 0
    total_tensors = 0
    metadata = {}

    # 2. Мгновенное чтение заголовков через safe_open (без загрузки весов в RAM)
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            # Читаем метаданные заголовка
            metadata = f.metadata() or {}
            
            # Считаем точное количество параметров
            for key in f.keys():
                shape = f.get_slice(key).get_shape()
                total_params += math.prod(shape)
                total_tensors += 1
    except Exception as e:
        console.print(t("[bold red]❌ Ошибка чтения Safetensors:[/bold red]") + f" {e}")
        return

    elapsed = time.time() - start_time
    size_gb = os.path.getsize(file_path) / (1024 ** 3)
    billions = total_params / 1_000_000_000

    # 3. Поиск триггерных слов и метаданных обучения (Kohya / ModelSpec / Civitai)
    trigger_words = []
    base_model = metadata.get("ss_sd_model_name", metadata.get("ss_base_model_version", t("Не указано")))
    
    # Вариант A: Поле ss_trained_words (Kohya)
    if "ss_trained_words" in metadata:
        try:
            words = json.loads(metadata["ss_trained_words"])
            if isinstance(words, list):
                trigger_words.extend(words)
            elif isinstance(words, str):
                trigger_words.append(words)
        except Exception:
            trigger_words.append(str(metadata["ss_trained_words"]))

    # Вариант B: Поле modelspec.trigger_phrase
    if "modelspec.trigger_phrase" in metadata:
        trigger_words.append(metadata["modelspec.trigger_phrase"])

    # Вариант C: Анализ частоты тегов (ss_tag_frequency)
    if "ss_tag_frequency" in metadata and not trigger_words:
        try:
            tag_freq = json.loads(metadata["ss_tag_frequency"])
            top_tags = []
            for dir_tags in tag_freq.values():
                sorted_tags = sorted(dir_tags.items(), key=lambda x: x[1], reverse=True)
                top_tags.extend([tag for tag, count in sorted_tags[:5]])
            if top_tags:
                trigger_words.append(", ".join(set(top_tags[:5])))
        except Exception:
            pass

    triggers_str = ", ".join(set(trigger_words)) if trigger_words else t("[dim yellow]Триггерные слова не найдены в метаданных[/dim yellow]")

    # 4. Вывод красивой Rich-таблицы
    table = Table(title=t("[bold green]📊 ИНСПЕКЦИЯ МОДЕЛИ:[/bold green]") + f" [bold green]{os.path.basename(file_path)}[/bold green]", show_header=True, header_style="bold cyan")
    table.add_column(t("Параметр"), style="white")
    table.add_column(t("Значение"), style="bold yellow")

    table.add_row(t("Файл"), f"[bold white]{os.path.basename(file_path)}[/bold white]")
    table.add_row(t("Размер на диске"), f"[bold green]{size_gb:.2f} GB[/bold green]")
    table.add_row(t("Всего слоёв / тензоров"), f"{total_tensors:,}")
    table.add_row(t("Точное число параметров"), f"{total_params:,}")
    table.add_row(t("Параметров в миллиардах"), f"[bold green]{billions:.6f} B[/bold green]")
    table.add_row(t("Базовая модель"), f"{base_model}")
    table.add_row(t("Триггерные слова"), f"[bold cyan]{triggers_str}[/bold cyan]")
    table.add_row(t("Время сканирования"), f"{elapsed:.4f} " + t("сек"))

    console.print("\n")
    console.print(table)

if __name__ == "__main__":
    inspect_model()