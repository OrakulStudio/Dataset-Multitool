# ==============================================================================
#  ORAKUL CORE — AI Dataset & Model MultiTool
#  Module: Dataset Manager
# ------------------------------------------------------------------------------
#  Author:      Orakul (Orakul Studio)
#  GitHub:      https://github.com/OrakulStudio
#  Civitai:     https://civitai.com/user/ORAKUL_STUDIO
#  HuggingFace: https://huggingface.co/OrakulStorm
# ------------------------------------------------------------------------------
#  Description: Clean single-purpose dataset tools.
#  Copyright (c) 2026 Orakul Studio. All rights reserved.
# ==============================================================================

import json
import os
import random
import re
import time
from pathlib import Path
from PIL import Image, ImageOps
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

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
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def clean_path(path_str: str) -> str:
    return path_str.strip(' "\'')


def get_num_digits_choice(total_items: int) -> int:
    auto_digits = max(4, len(str(total_items)))
    console.print("\n" + t("[bold white]Формат нумерации файлов:[/bold white]"))
    
    text_part1 = t("[bold cyan][1][/bold cyan] Автовыбор (для ")
    text_part2 = t(" файлов: [bold yellow]")
    text_part3 = t("[/bold yellow] знака, например [bold green]")
    text_part4 = t("[/bold green])")
    
    console.print(f"{text_part1}{total_items}{text_part2}{auto_digits}{text_part3}{1:0{auto_digits}d}{text_part4}")
    console.print(t("[bold cyan][2][/bold cyan] 2 знака ([bold green]01, 02...[/bold green])"))
    console.print(t("[bold cyan][3][/bold cyan] 3 знака ([bold green]001, 002...[/bold green])"))
    console.print(t("[bold cyan][4][/bold cyan] 4 знака ([bold green]0001, 0002...[/bold green])"))
    console.print(t("[bold cyan][5][/bold cyan] 5 знаков ([bold green]00001, 00002...[/bold green])"))

    choice = console.input("\n" + t("[bold cyan]Выбери вариант (1-5) [По умолчанию: 1]: [/bold cyan]")).strip()
    if choice == '2': return 2
    if choice == '3': return 3
    if choice == '4': return 4
    if choice == '5': return 5
    return auto_digits


def _show_summary(title: str, originals: int, total_created: int, output_path: str, elapsed: float):
    summary = Table(title=f"[bold green]{t('📊 ОТЧЕТ:')} {title}[/bold green]", show_header=True, header_style="bold cyan")
    summary.add_column(t("Параметр"), style="white")
    summary.add_column(t("Значение"), style="bold yellow")

    summary.add_row(t("Исходных файлов"), f"{originals}")
    summary.add_row(t("Создано итоговых пар"), f"[bold green]{total_created}[/bold green]")
    summary.add_row(t("Сохранено в папку"), f"[bold white]{output_path}[/bold white]")
    summary.add_row(t("Затрачено времени"), f"{elapsed:.2f} {t('сек')}")

    console.print("\n")
    console.print(summary)


# =========================================================================
# 1. CROP & RESIZE (только кроп + ресайз, имена не трогаем)
# =========================================================================
def crop_and_resize_ui():
    console.print(Panel(t("[bold yellow]✂️ CROP & RESIZE • Квадратный кроп + ресайз[/bold yellow]"), expand=False))

    input_folder = clean_path(console.input("\n" + t("[bold cyan]1. Перетащи исходную папку с датасетом: [/bold cyan]")))
    if not os.path.exists(input_folder) or not os.path.isdir(input_folder):
        console.print(t("[bold red]❌ Ошибка: Папка отсутствует![/bold red]"))
        return

    output_folder = clean_path(console.input(t("[bold cyan]2. Перетащи или укажи папку сохранения: [/bold cyan]")))
    if not output_folder:
        output_folder = str(Path(input_folder) / "cropped_resized")

    size_str = console.input(t("[bold cyan]3. Размер стороны квадрата (например 1024) [По умолчанию 1024]: [/bold cyan]")).strip()
    size = int(size_str) if size_str.isdigit() else 1024

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_folder)

    images = [f for f in input_path.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
    if not images:
        console.print(t("[bold red]❌ В папке не найдено подходящих изображений![/bold red]"))
        return

    start_time = time.time()

    with Progress(
        SpinnerColumn("dots", style="bold yellow"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=35, style="black", complete_style="bold green", finished_style="bold green"),
        TaskProgressColumn("[bold yellow]{task.percentage:>3.0f}%"),
        TextColumn(t("• [bold magenta]{task.completed}/{task.total}[/] файлов")),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(t("Кроп и ресайз"), total=len(images))

        for img_file in images:
            try:
                with Image.open(img_file) as img:
                    width, height = img.size
                    min_dim = min(width, height)
                    left = (width - min_dim) / 2
                    top = (height - min_dim) / 2
                    right = (width + min_dim) / 2
                    bottom = (height + min_dim) / 2

                    img_cropped = img.crop((left, top, right, bottom))
                    img_resized = img_cropped.resize((size, size), Image.LANCZOS)

                    out_img = output_path / img_file.name
                    img_resized.save(out_img, quality=95)

                    # Копируем парный .txt если есть
                    txt_file = img_file.with_suffix('.txt')
                    if txt_file.exists():
                        out_txt = output_path / txt_file.name
                        out_txt.write_text(txt_file.read_text(encoding='utf-8'), encoding='utf-8')

            except Exception as e:
                console.print(f"\n[yellow]⚠ {img_file.name}: {e}[/yellow]")

            progress.update(task, advance=1)

    elapsed = time.time() - start_time
    _show_summary("CROP & RESIZE", len(images), len(images), output_folder, elapsed)


# =========================================================================
# 2. QUICK SHUFFLE (только перемешивание прямо в указанной папке)
# =========================================================================
def quick_shuffle_ui():
    console.print(Panel(t("[bold yellow]🎲 QUICK SHUFFLE • Перемешивание порядка (на месте)[/bold yellow]"), expand=False))

    input_folder = clean_path(console.input("\n" + t("[bold cyan]1. Перетащи папку с датасетом (файлы будут перемешаны прямо здесь): [/bold cyan]")))
    if not os.path.exists(input_folder) or not os.path.isdir(input_folder):
        console.print(t("[bold red]❌ Ошибка: Папка отсутствует![/bold red]"))
        return

    confirm = console.input(t("[bold yellow]Файлы будут переименованы в этой же папке. Продолжить? (y/n): [/bold yellow]")).strip().lower()
    if confirm not in ('y', 'yes', 'д', 'да'):
        console.print(t("[bold yellow]Отменено.[/bold yellow]"))
        return

    input_path = Path(input_folder)

    items = []
    for img_file in input_path.iterdir():
        if img_file.suffix.lower() in VALID_EXTENSIONS:
            txt_file = img_file.with_suffix('.txt')
            items.append({
                'img_path': img_file,
                'txt_path': txt_file if txt_file.exists() else None,
                'ext': img_file.suffix.lower()
            })

    if not items:
        console.print(t("[bold red]❌ В папке не найдено подходящих изображений![/bold red]"))
        return

    num_digits = get_num_digits_choice(len(items))
    random.shuffle(items)

    # Сначала во временные имена, чтобы избежать конфликтов
    temp_pairs = []
    for i, item in enumerate(items):
        temp_id = f"__temp_shuffle_{i}"
        temp_img = input_path / f"{temp_id}{item['ext']}"
        item['img_path'].rename(temp_img)

        temp_txt = None
        if item['txt_path'] and item['txt_path'].exists():
            temp_txt = input_path / f"{temp_id}.txt"
            item['txt_path'].rename(temp_txt)

        temp_pairs.append((temp_img, temp_txt, item['ext']))

    start_time = time.time()

    with Progress(
        SpinnerColumn("dots", style="bold yellow"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=35, style="black", complete_style="bold green", finished_style="bold green"),
        TaskProgressColumn("[bold yellow]{task.percentage:>3.0f}%"),
        TextColumn(t("• [bold magenta]{task.completed}/{task.total}[/] файлов")),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(t("Перемешивание"), total=len(temp_pairs))

        for idx, (temp_img, temp_txt, ext) in enumerate(temp_pairs, start=1):
            new_name = f"{idx:0{num_digits}d}"
            final_img = input_path / f"{new_name}{ext}"
            temp_img.rename(final_img)

            if temp_txt and temp_txt.exists():
                final_txt = input_path / f"{new_name}.txt"
                temp_txt.rename(final_txt)

            progress.update(task, advance=1)

    elapsed = time.time() - start_time
    _show_summary("QUICK SHUFFLE", len(items), len(items), input_folder, elapsed)


# =========================================================================
# 3. MIRROR FLIP (только зеркалирование на месте, без дубликатов)
# =========================================================================
def mirror_flip_ui():
    console.print(Panel(t("[bold yellow]🪞 MIRROR FLIP • Зеркалирование на месте[/bold yellow]"), expand=False))

    input_folder = clean_path(console.input("\n" + t("[bold cyan]1. Перетащи папку с датасетом (файлы будут перезаписаны!): [/bold cyan]")))
    if not os.path.exists(input_folder) or not os.path.isdir(input_folder):
        console.print(t("[bold red]❌ Ошибка: Папка отсутствует![/bold red]"))
        return

    confirm = console.input(t("[bold red]ВНИМАНИЕ! Оригиналы будут перезаписаны. Продолжить? (y/n): [/bold red]")).strip().lower()
    if confirm not in ('y', 'yes', 'д', 'да'):
        console.print(t("[bold yellow]Отменено.[/bold yellow]"))
        return

    input_path = Path(input_folder)
    images = [f for f in input_path.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]

    if not images:
        console.print(t("[bold red]❌ В папке не найдено подходящих изображений![/bold red]"))
        return

    start_time = time.time()

    with Progress(
        SpinnerColumn("dots", style="bold yellow"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=35, style="black", complete_style="bold green", finished_style="bold green"),
        TaskProgressColumn("[bold yellow]{task.percentage:>3.0f}%"),
        TextColumn(t("• [bold magenta]{task.completed}/{task.total}[/] файлов")),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(t("Зеркалирование"), total=len(images))

        for img_file in images:
            try:
                with Image.open(img_file) as img:
                    img = ImageOps.mirror(img)
                    img.save(img_file)

                # Меняем left <-> right в тексте
                txt_file = img_file.with_suffix('.txt')
                if txt_file.exists():
                    try:
                        caption = txt_file.read_text(encoding='utf-8')
                        caption = re.sub(r'\bleft\b', '___TEMP_LEFT___', caption, flags=re.IGNORECASE)
                        caption = re.sub(r'\bright\b', 'left', caption, flags=re.IGNORECASE)
                        caption = re.sub(r'___TEMP_LEFT___', 'right', caption)
                        txt_file.write_text(caption, encoding='utf-8')
                    except Exception:
                        pass

            except Exception as e:
                console.print(f"\n[yellow]⚠ {img_file.name}: {e}[/yellow]")

            progress.update(task, advance=1)

    elapsed = time.time() - start_time
    _show_summary("MIRROR FLIP", len(images), len(images), input_folder, elapsed)


# =========================================================================
# 4. DATASET MULTIPLIER (классическая аугментация — создаёт дубликаты)
# =========================================================================
def dataset_multiplier_ui():
    console.print(Panel(t("[bold yellow]📈 DATASET MULTIPLIER • Аугментация (оригинал + зеркало)[/bold yellow]"), expand=False))

    input_folder = clean_path(console.input("\n" + t("[bold cyan]1. Перетащи исходную папку с датасетом: [/bold cyan]")))
    if not os.path.exists(input_folder) or not os.path.isdir(input_folder):
        console.print(t("[bold red]❌ Ошибка: Папка отсутствует![/bold red]"))
        return

    output_folder = clean_path(console.input(t("[bold cyan]2. Перетащи или укажи папку сохранения: [/bold cyan]")))
    if not output_folder:
        output_folder = str(Path(input_folder) / "augmented_dataset")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_folder)

    items = []
    for img_file in input_path.iterdir():
        if img_file.suffix.lower() in VALID_EXTENSIONS:
            txt_file = img_file.with_suffix('.txt')
            caption = ""
            if txt_file.exists():
                try:
                    caption = txt_file.read_text(encoding='utf-8')
                except Exception:
                    pass

            # Оригинал
            items.append({
                'img_path': img_file,
                'caption': caption,
                'is_flipped': False,
                'ext': img_file.suffix.lower()
            })
            # Зеркало
            items.append({
                'img_path': img_file,
                'caption': caption,
                'is_flipped': True,
                'ext': img_file.suffix.lower()
            })

    if not items:
        console.print(t("[bold red]❌ В папке не найдено подходящих изображений![/bold red]"))
        return

    num_digits = get_num_digits_choice(len(items))
    random.shuffle(items)

    start_time = time.time()

    with Progress(
        SpinnerColumn("dots", style="bold yellow"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=35, style="black", complete_style="bold green", finished_style="bold green"),
        TaskProgressColumn("[bold yellow]{task.percentage:>3.0f}%"),
        TextColumn(t("• [bold magenta]{task.completed}/{task.total}[/] файлов")),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(t("Аугментация"), total=len(items))

        for idx, item in enumerate(items, start=1):
            new_name = f"{idx:0{num_digits}d}"
            out_img = output_path / f"{new_name}{item['ext']}"
            out_txt = output_path / f"{new_name}.txt"

            try:
                with Image.open(item['img_path']) as img:
                    if item['is_flipped']:
                        img = ImageOps.mirror(img)
                    img.save(out_img)
            except Exception as e:
                console.print(f"\n[yellow]⚠ {item['img_path'].name}: {e}[/yellow]")

            caption_text = item['caption']
            if item['is_flipped'] and caption_text:
                caption_text = re.sub(r'\bleft\b', '___TEMP_LEFT___', caption_text, flags=re.IGNORECASE)
                caption_text = re.sub(r'\bright\b', 'left', caption_text, flags=re.IGNORECASE)
                caption_text = re.sub(r'___TEMP_LEFT___', 'right', caption_text)

            if caption_text:
                try:
                    out_txt.write_text(caption_text, encoding='utf-8')
                except Exception:
                    pass

            progress.update(task, advance=1)

    elapsed = time.time() - start_time
    _show_summary("DATASET MULTIPLIER", len(items)//2, len(items), output_folder, elapsed)


# =========================================================================
# 5. BATCH RENAME (чистое переименование)
# =========================================================================
def batch_rename_ui():
    console.print(Panel(t("[bold yellow]🏷 BATCH RENAME • Чистое переименование[/bold yellow]"), expand=False))

    input_folder = clean_path(console.input("\n" + t("[bold cyan]1. Перетащи папку с датасетом: [/bold cyan]")))
    if not os.path.exists(input_folder) or not os.path.isdir(input_folder):
        console.print(t("[bold red]❌ Ошибка: Папка отсутствует![/bold red]"))
        return

    prefix = console.input(t("[bold cyan]2. Префикс (Enter = без префикса): [/bold cyan]")).strip()

    input_path = Path(input_folder)
    images = [f for f in input_path.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
    if not images:
        console.print(t("[bold red]❌ В папке не найдено подходящих изображений![/bold red]"))
        return

    images.sort(key=lambda x: x.name.lower())
    num_digits = get_num_digits_choice(len(images))

    # Сначала во временные имена, чтобы избежать конфликтов
    temp_files = []
    for i, img_file in enumerate(images):
        temp_name = f"__temp_rename_{i}{img_file.suffix}"
        temp_path = img_file.parent / temp_name
        img_file.rename(temp_path)

        txt_file = img_file.with_suffix('.txt')
        temp_txt = None
        if txt_file.exists():
            temp_txt = img_file.parent / f"__temp_rename_{i}.txt"
            txt_file.rename(temp_txt)

        temp_files.append((temp_path, temp_txt, img_file.suffix))

    start_time = time.time()

    with Progress(
        SpinnerColumn("dots", style="bold yellow"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=35, style="black", complete_style="bold green", finished_style="bold green"),
        TaskProgressColumn("[bold yellow]{task.percentage:>3.0f}%"),
        TextColumn(t("• [bold magenta]{task.completed}/{task.total}[/] файлов")),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(t("Переименование"), total=len(temp_files))

        for idx, (temp_img, temp_txt, ext) in enumerate(temp_files, start=1):
            new_name = f"{prefix}{idx:0{num_digits}d}" if prefix else f"{idx:0{num_digits}d}"
            final_img = temp_img.parent / f"{new_name}{ext}"
            temp_img.rename(final_img)

            if temp_txt and temp_txt.exists():
                final_txt = temp_txt.parent / f"{new_name}.txt"
                temp_txt.rename(final_txt)

            progress.update(task, advance=1)

    elapsed = time.time() - start_time
    _show_summary("BATCH RENAME", len(images), len(images), input_folder, elapsed)


# =========================================================================
# ГЛАВНОЕ МЕНЮ
# =========================================================================
def main():
    while True:
        menu_text = (
            t("[bold cyan][1][/bold cyan] Crop & Resize (Квадратный кроп + ресайз, имена сохраняются)") + "\n" +
            t("[bold cyan][2][/bold cyan] Quick Shuffle (Только перемешивание порядка)") + "\n" +
            t("[bold cyan][3][/bold cyan] Mirror Flip (Зеркалирование на месте, без дубликатов)") + "\n" +
            t("[bold cyan][4][/bold cyan] Dataset Multiplier (Оригинал + зеркало → новая папка)") + "\n" +
            t("[bold cyan][5][/bold cyan] Batch Rename (Чистое переименование)") + "\n" +
            t("[bold red][0][/bold red] Выход в главное меню")
        )
        console.print(Panel(menu_text, title=t("[bold yellow]⚡ ORAKUL STUDIO • DATASET TOOLKIT ⚡[/bold yellow]"), border_style="yellow", expand=False))

        choice = console.input("\n" + t("[bold white]Выбери режим (0-5): [/bold white]")).strip()

        if choice == '1':
            crop_and_resize_ui()
            break
        elif choice == '2':
            quick_shuffle_ui()
            break
        elif choice == '3':
            mirror_flip_ui()
            break
        elif choice == '4':
            dataset_multiplier_ui()
            break
        elif choice == '5':
            batch_rename_ui()
            break
        elif choice == '0':
            break


if __name__ == "__main__":
    main()