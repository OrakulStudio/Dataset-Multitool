#!/usr/bin/env python3
"""
Мощный совмещённый модуль: Умный квадратный кроп + рескейл + моментальная нумерация с прогресс-баром.
"""

import json
import os
import re
import sys
import shutil
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

def process_dataset(input_dir: Path, output_dir: Path, size: int = 1024, prefix: str = ""):
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    
    if not input_dir.exists():
        console.print(t("[bold red][-] Ошибка: Исходная папка не найдена: ") + str(input_dir) + t("[/bold red]"))
        return

   # Находим только изображения
    files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    files.sort(key=natural_sort_key)

    if not files:
        console.print(f"[bold yellow][!] {t('В папке')} {input_dir} {t('не найдено подходящих изображений.')}[/bold yellow]")
        return

    # Сканируем на наличие строго парных .txt файлов (игнорируя левый кэш)
    matched_txt_files = [f.with_suffix('.txt') for f in files if f.with_suffix('.txt').exists()]
    
    sync_txt = False
    if matched_txt_files:
        console.print(f"\n[bold yellow]📄 {t('Обнаружено')} {len(matched_txt_files)} {t('парных текстовых файлов (.txt)!')}[/bold yellow]")
        txt_choice = console.input(f"[bold cyan]{t('Перенести и переименовать их вместе с картинками? [Y/n]:')} [/bold cyan]").strip().lower()
        if txt_choice in ['y', 'yes', '']:
            sync_txt = True

    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[bold cyan][*] {t('Найдено файлов для обработки:')}[/bold cyan] {len(files)}")
    prefix_display = prefix + "_XXXXX" if prefix else t("Оригинальные имена")
    console.print(f"[bold cyan][*] {t('Целевой размер:')}[/bold cyan] {size}x{size} | [cyan]{t('Префикс:')}[/cyan] {prefix_display}\n")

    success_count = 0
    logs = []

    # Используем живой прогресс-бар
    for i, file_path in enumerate(track(files, description=f"[bold yellow]{t('Обработка и ресайз кадров...')}[/bold yellow]"), start=1):
        ext = file_path.suffix.lower()
        original_stem = file_path.stem
        
        # Логика именования
        if prefix:
            new_stem = f"{prefix}_{i:05d}"
        else:
            new_stem = original_stem
            
        new_filename = f"{new_stem}{ext}"
        dest_path = output_dir / new_filename

        try:
            # Обработка картинки
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
                logs.append(f"[white]• {file_path.name}[/white] ➔ [green]{new_filename}[/green]")
                
            # Синхронизация парного текстового файла
            src_txt = file_path.with_suffix('.txt')
            if sync_txt and src_txt.exists():
                dest_txt = output_dir / f"{new_stem}.txt"
                shutil.copy2(src_txt, dest_txt)
                
        except Exception as e:
            logs.append(f"[bold red]• {file_path.name} ➔ {t('Ошибка:')} {e}[/bold red]")

    # Красивый итоговый отчёт
    console.print(Panel(
        "\n".join(logs[-15:] if len(logs) > 15 else logs),  
        title=f"[bold green]✨ {t('ОБРАБОТКА ЗАВЕРШЕНА (Успешно:')} {success_count}/{len(files)})[/bold green]",
        border_style="green",
        expand=False
    ))
    console.print(f"[bold cyan][+] {t('Готовый датасет сохранён в:')}[/bold cyan] {output_dir}")

def main():
    if len(sys.argv) > 1:
        base_folder = Path(sys.argv[1]).resolve()
    else:
        base_folder = Path(console.input(f"[bold cyan]{t('Перетащи папку проекта датасета сюда:')} [/bold cyan]").strip(' "\'')).resolve()

    if not base_folder.exists():
        console.print(f"[bold red][-] {t('Указанный путь не существует!')}[/bold red]")
        return

    input_dir = base_folder / "raw" if (base_folder / "raw").exists() else base_folder
    output_dir = base_folder / "ready"

    size_input = console.input(f"[bold yellow]{t('Размер квадрата в пикселях [Enter = 1024]:')} [/bold yellow]").strip()
    size = int(size_input) if size_input.isdigit() else 1024

    # Префикс теперь можно оставить пустым
    prefix = console.input(f"[bold yellow]{t('Префикс для файлов [Enter = оставить родные имена]:')} [/bold yellow]").strip()

    process_dataset(input_dir, output_dir, size, prefix)

if __name__ == "__main__":
    main()