# ==============================================================================
#  ORAKUL CORE — AI Dataset & Model MultiTool
#  Module: Model Dataset 
# ------------------------------------------------------------------------------
#  Author:      Orakul (Orakul Studio)
#  GitHub:      https://github.com/OrakulStudio
#  Civitai:     https://civitai.com/user/ORAKUL_STUDIO
#  HuggingFace: https://huggingface.co/OrakulStorm
# ------------------------------------------------------------------------------
#  Description: Fast safetensors header reading, rank & parameter counter.
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
    """Очистка пути от кавычек и лишних пробелов при Drag-and-Drop"""
    return path_str.strip(' "\'')

def get_num_digits_choice(total_items: int) -> int:
    """Выбор формата нумерации (количества нулей в префиксе)"""
    auto_digits = max(4, len(str(total_items)))
    console.print("\n" + t("[bold white]Формат нумерации файлов:[/bold white]"))
    
    # Распил f-строки
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


# =========================================================================
# 1. СЛЕВА НАПРАВО (Аугментация + Отражение + Auto-Swap left/right + Shuffle)
# =========================================================================
def augment_left_right_ui():
    console.print(Panel(t("[bold yellow]🖼 DATASET AUGMENTOR • СЛЕВА НАПРАВО + SWAP LEFT/RIGHT 🖼[/bold yellow]"), expand=False))

    input_folder = clean_path(console.input("\n" + t("[bold cyan]1. Перетащи исходную папку с датасетом: [/bold cyan]")))
    if not os.path.exists(input_folder) or not os.path.isdir(input_folder):
        console.print(t("[bold red]❌ Ошибка: Папка отсутствует![/bold red]"))
        return

    output_folder = clean_path(console.input(t("[bold cyan]2. Перетащи или укажи папку сохранения готового датасета: [/bold cyan]")))
    if not output_folder:
        output_folder = str(Path(input_folder) / "augmented_dataset")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_folder)
    items = []

    # Сбор пар картинка + .txt
    for img_file in input_path.iterdir():
        if img_file.suffix.lower() in VALID_EXTENSIONS:
            txt_file = img_file.with_suffix('.txt')
            caption = ""
            if txt_file.exists():
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        caption = f.read()
                except Exception:
                    pass

            # Оригинал
            items.append({
                'img_path': img_file,
                'caption': caption,
                'is_flipped': False,
                'ext': img_file.suffix.lower()
            })

            # Отражённый дубликат
            items.append({
                'img_path': img_file,
                'caption': caption,
                'is_flipped': True,
                'ext': img_file.suffix.lower()
            })

    if not items:
        console.print(t("[bold red]❌ В папке не найдено подходящих изображений![/bold red]"))
        return

    text_part1 = t("[bold green]✓[/bold green] Найдено исходников: [bold cyan]")
    text_part2 = t("[/bold cyan] | Будет сгенерировано с зеркалами: [bold yellow]")
    text_part3 = t("[/bold yellow]")
    console.print("\n" + f"{text_part1}{len(items)//2}{text_part2}{len(items)}{text_part3}")

    num_digits = get_num_digits_choice(len(items))

    # Рандомное перемешивание
    random.shuffle(items)

    start_time = time.time()
    processed_count = 0

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
        task = progress.add_task(t("Аугментация датасета"), total=len(items))

        for idx, item in enumerate(items, start=1):
            new_name = f"{idx:0{num_digits}d}"
            out_img = output_path / f"{new_name}{item['ext']}"
            out_txt = output_path / f"{new_name}.txt"

            # 1. Картинка
            try:
                with Image.open(item['img_path']) as img:
                    if item['is_flipped']:
                        img = ImageOps.mirror(img)
                    img.save(out_img)
            except Exception as e:
                console.print("\n" + t("[yellow]⚠ Ошибка сохранения картинки [/yellow]") + f"[yellow]{item['img_path'].name}: {e}[/yellow]")

            # 2. Текст с авто-заменой left <-> right
            caption_text = item['caption']
            if item['is_flipped'] and caption_text:
                caption_text = re.sub(r'\bleft\b', '___TEMP_LEFT___', caption_text, flags=re.IGNORECASE)
                caption_text = re.sub(r'\bright\b', 'left', caption_text, flags=re.IGNORECASE)
                caption_text = re.sub(r'___TEMP_LEFT___', 'right', caption_text)

            if caption_text:
                try:
                    with open(out_txt, 'w', encoding='utf-8') as f:
                        f.write(caption_text)
                except Exception:
                    pass

            processed_count += 1
            progress.update(task, advance=1)

    elapsed = time.time() - start_time
    _show_summary("АУГМЕНТАЦИЯ (СЛЕВА НАПРАВО)", len(items)//2, len(items), output_folder, elapsed)


# =========================================================================
# 2. ПЕРЕМЕШИВАЛКА & НУМЕРАТОР (Shuffle & Renumber без отзеркаливания)
# =========================================================================
def shuffle_and_renumber_ui():
    console.print(Panel(t("[bold yellow]🎲 ПЕРЕМЕШИВАЛКА & НУМЕРАТОР ДАТАСЕТА 🎲[/bold yellow]"), expand=False))

    input_folder = clean_path(console.input("\n" + t("[bold cyan]1. Перетащи исходную папку с датасетом: [/bold cyan]")))
    if not os.path.exists(input_folder) or not os.path.isdir(input_folder):
        console.print(t("[bold red]❌ Ошибка: Папка нет не найдена![/bold red]"))
        return

    output_folder = clean_path(console.input(t("[bold cyan]2. Перетащи или укажи папку сохранения (или Enter для сохранения рядом): [/bold cyan]")))
    if not output_folder:
        output_folder = str(Path(input_folder) / "shuffled_dataset")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_folder)
    items = []

    # Сбор пар картинка + .txt
    for img_file in input_path.iterdir():
        if img_file.suffix.lower() in VALID_EXTENSIONS:
            txt_file = img_file.with_suffix('.txt')
            caption = ""
            if txt_file.exists():
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        caption = f.read()
                except Exception:
                    pass

            items.append({
                'img_path': img_file,
                'caption': caption,
                'ext': img_file.suffix.lower()
            })

    if not items:
        console.print(t("[bold red]❌ В папке не найдено подходящих изображений![/bold red]"))
        return

    text_part1 = t("[bold green]✓[/bold green] Найдено файлов для перемешивания: [bold cyan]")
    text_part2 = t("[/bold cyan]")
    console.print("\n" + f"{text_part1}{len(items)}{text_part2}")

    num_digits = get_num_digits_choice(len(items))

    # Рандомное перемешивание
    random.shuffle(items)

    start_time = time.time()
    processed_count = 0

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
        task = progress.add_task(t("Перемешивание и нумерация"), total=len(items))

        for idx, item in enumerate(items, start=1):
            new_name = f"{idx:0{num_digits}d}"
            out_img = output_path / f"{new_name}{item['ext']}"
            out_txt = output_path / f"{new_name}.txt"

            # Копирование / Перезапись изображения
            try:
                with Image.open(item['img_path']) as img:
                    img.save(out_img)
            except Exception as e:
                console.print("\n" + t("[yellow]⚠ Ошибка записи [/yellow]") + f"[yellow]{item['img_path'].name}: {e}[/yellow]")

            # Сохранение .txt описания
            if item['caption']:
                try:
                    with open(out_txt, 'w', encoding='utf-8') as f:
                        f.write(item['caption'])
                except Exception:
                    pass

            processed_count += 1
            progress.update(task, advance=1)

    elapsed = time.time() - start_time
    _show_summary("ПЕРЕМЕШИВАЛКА & НУМЕРАЦИЯ", len(items), len(items), output_folder, elapsed)


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
# ГЛАВНОЕ МЕНЮ ОБРАБОТКИ ДАТАСЕТОВ
# =========================================================================
def main():
    while True:
        menu_text = (
            t("[bold cyan][1][/bold cyan] Слева направо (Дублирование + Зеркало + Auto-Swap left/right + Shuffle)") + "\n" +
            t("[bold cyan][2][/bold cyan] Перемешивалка & Нумератор (Перемешать датасет и задать префикс 01/001/0001)") + "\n" +
            t("[bold red][0][/bold red] Выход в главное меню")
        )
        console.print(Panel(menu_text, title=t("[bold yellow]⚡ ORAKUL STUDIO • DATASET TOOLKIT ⚡[/bold yellow]"), border_style="yellow", expand=False))

        choice = console.input("\n" + t("[bold white]Выбери режим (0-2): [/bold white]")).strip()

        if choice == '1':
            augment_left_right_ui()
            break
        elif choice == '2':
            shuffle_and_renumber_ui()
            break
        elif choice == '0':
            break

if __name__ == "__main__":
    main()