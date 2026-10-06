#!/usr/bin/env python3
"""
Мощный совмещённый модуль: Умный квадратный кроп + рескейл + моментальная нумерация с прогресс-баром.
"""

import json
import os
import re
import sys
from pathlib import Path
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

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

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

def process_dataset(input_dir: Path, output_dir: Path, size: int = 1024, prefix: str = "rt"):
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    
    if not input_dir.exists():
        console.print(t("[bold red][-] Ошибка: Исходная папка не найдена: ") + str(input_dir) + t("[/bold red]"))
        return

    files = [f for f in input_dir.iterdir() if f.is_file() and f.name.lower().endswith(image_extensions)]
    files.sort(key=natural_sort_key)

    if not files:
        console.print(t("[bold yellow][!] В папке ") + str(input_dir) + t(" не найдено подходящих изображений.[/bold yellow]"))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(t("[bold cyan][*] Найдено файлов для обработки:[/bold cyan] ") + str(len(files)))
    console.print(t("[bold cyan][*] Целевой размер:[/bold cyan] ") + str(size) + t("x") + str(size) + t(" | [cyan]Префикс:[/cyan] ") + prefix + "_XXXXX\n")

    success_count = 0
    logs = []

    # Используем живой прогресс-бар rich вместо мерцающего курсора
    for i, file_path in enumerate(track(files, description=t("[bold yellow]Обработка и ресайз кадров...[/bold yellow]")), start=1):
        ext = file_path.suffix.lower()
        new_filename = f"{prefix}_{i:05d}{ext}"
        dest_path = output_dir / new_filename

        try:
            with Image.open(file_path) as img:
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                    
                width, height = img.size
                min_dim = min(width, height)
                
                left = (width - min_dim) / 2
                top = (height - min_dim) / 2
                right = (width + min_dim) / 2
                bottom = (height + min_dim) / 2
                
                img_cropped = img.crop((left, top, right, bottom))
                img_resized = img_cropped.resize((size, size), Image.LANCZOS)
                
                img_resized.save(dest_path, quality=95)
                success_count += 1
                logs.append(t("[white]• ") + file_path.name + t("[/white] ➔ [green]") + new_filename + t("[/green]"))
        except Exception as e:
            logs.append(t("[bold red]• ") + file_path.name + t(" ➔ Ошибка: ") + str(e) + t("[/bold red]"))

    # Красивый итоговый отчёт
    console.print(Panel(
        "\n".join(logs[-15:] if len(logs) > 15 else logs),  # Показываем последние 15 для компактности, если файлов много
        title=t("[bold green]✨ ОБРАБОТКА ЗАВЕРШЕНА (Успешно: ") + str(success_count) + "/" + str(len(files)) + t(")[/bold green]"),
        border_style="green",
        expand=False
    ))
    console.print(t("[bold cyan][+] Готовый датасет сохранён в:[/bold cyan] ") + str(output_dir))

def main():
    if len(sys.argv) > 1:
        base_folder = Path(sys.argv[1]).resolve()
    else:
        base_folder = Path(console.input(t("[bold cyan]Перетащи папку проекта датасета сюда:[/bold cyan] ")).strip(' "\'')).resolve()

    if not base_folder.exists():
        console.print(t("[bold red][-] Указанный путь не существует![/bold red]"))
        return

    input_dir = base_folder / "raw" if (base_folder / "raw").exists() else base_folder
    output_dir = base_folder / "ready"

    size_input = console.input(t("[bold yellow]Размер квадрата в пикселях [Enter = 1024]: [/bold yellow]")).strip()
    size = int(size_input) if size_input.isdigit() else 1024

    prefix_input = console.input(t("[bold yellow]Префикс для файлов [Enter = rt]: [/bold yellow]")).strip()
    prefix = prefix_input if prefix_input else "rt"

    process_dataset(input_dir, output_dir, size, prefix)

if __name__ == "__main__":
    main()