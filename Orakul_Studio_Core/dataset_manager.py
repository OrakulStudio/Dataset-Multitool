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
from rich import box
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

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
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

def clean_path(path_str: str) -> str:
    return path_str.strip(' "\'')

def get_num_digits_choice(total_items: int) -> int:
    auto_digits = max(4, len(str(total_items)))
    console.print("\n" + f"[bold white]── {t('File numbering format:')} ──[/bold white]")
    
    console.print(f"  [bold white][ 1 ][/bold white] [white]{t('Auto-detect')}[/white] [dim]({t('for')} {total_items} {t('files:')} {auto_digits} {t('digits, e.g.')} {1:0{auto_digits}d})[/dim]")
    console.print(f"  [bold white][ 2 ][/bold white] [white]2 {t('digits')}[/white] [dim](01, 02...)[/dim]")
    console.print(f"  [bold white][ 3 ][/bold white] [white]3 {t('digits')}[/white] [dim](001, 002...)[/dim]")
    console.print(f"  [bold white][ 4 ][/bold white] [white]4 {t('digits')}[/white] [dim](0001, 0002...)[/dim]")
    console.print(f"  [bold white][ 5 ][/bold white] [white]5 {t('digits')}[/white] [dim](00001, 00002...)[/dim]")

    choice = console.input(f"\n[bold white]> {t('Select option')} (1-5) [[dim]{t('Default:')} 1[/dim]]: [/bold white]").strip()
    if choice == '2': return 2
    if choice == '3': return 3
    if choice == '4': return 4
    if choice == '5': return 5
    return auto_digits

def _show_summary(title: str, originals: int, total_created: int, output_path: str, elapsed: float):
    summary = Table(
        title=f"[bold white]REPORT: {title}[/bold white]",
        show_header=True,
        header_style="bold white",
        border_style="grey50",
        box=box.ROUNDED
    )
    summary.add_column(t("Parameter"), style="dim")
    summary.add_column(t("Value"), style="white")

    summary.add_row(t("Original files"), f"{originals}")
    summary.add_row(t("Total items created"), f"[bold white]{total_created}[/bold white]")
    summary.add_row(t("Saved to folder"), f"[white]{output_path}[/white]")
    summary.add_row(t("Time elapsed"), f"{elapsed:.2f} {t('sec')}")

    console.print("\n")
    console.print(summary)

def init_progress(task_desc: str, total: int):
    progress = Progress(
        SpinnerColumn("dots", style="white"),
        TextColumn("[bold white]{task.description}"),
        BarColumn(bar_width=35, style="grey50", complete_style="white", finished_style="bold white"),
        TaskProgressColumn("[white]{task.percentage:>3.0f}%"),
        TextColumn(f"[dim]•[/dim] [white]{{task.completed}}/{{task.total}}[/white] [dim]{t('files')}[/dim]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    task = progress.add_task(task_desc, total=total)
    return progress, task


# =========================================================================
# 1. CROP & RESIZE
# =========================================================================
def crop_and_resize_ui():
    console.print(Panel(
        f"[bold white]CROP & RESIZE[/bold white] [dim]• {t('Square crop + resize')}[/dim]",
        border_style="grey50",
        box=box.ROUNDED,
        expand=False
    ))

    input_folder = clean_path(console.input(f"\n[bold white]> 1. {t('Drag and drop source dataset folder')}: [/bold white]"))
    if not os.path.exists(input_folder) or not os.path.isdir(input_folder):
        console.print(f"[bold white]\\[ERROR][/bold white] [dim]{t('Folder not found!')}[/dim]")
        return

    output_folder = clean_path(console.input(f"[bold white]> 2. {t('Drag and drop or specify output folder')}: [/bold white]"))
    if not output_folder:
        output_folder = str(Path(input_folder) / "cropped_resized")

    size_str = console.input(f"[bold white]> 3. {t('Square side size (e.g. 1024)')} [[dim]{t('Default: 1024')}[/dim]]: [/bold white]").strip()
    size = int(size_str) if size_str.isdigit() else 1024

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_folder)

    images = [f for f in input_path.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
    if not images:
        console.print(f"[bold white]\\[ERROR][/bold white] [dim]{t('No suitable images found in the folder!')}[/dim]")
        return

    start_time = time.time()
    console.print(f"\n[bold white]\\[SYSTEM][/bold white] [dim]{t('Starting process')}...[/dim]")

    progress, task = init_progress(t("Cropping and resizing"), len(images))
    with progress:
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

                    txt_file = img_file.with_suffix('.txt')
                    if txt_file.exists():
                        out_txt = output_path / txt_file.name
                        out_txt.write_text(txt_file.read_text(encoding='utf-8'), encoding='utf-8')
            except Exception as e:
                console.print(f"\n[bold white]\\[WARNING][/bold white] [dim]{img_file.name}: {e}[/dim]")
            progress.update(task, advance=1)

    _show_summary("CROP & RESIZE", len(images), len(images), output_folder, time.time() - start_time)


# =========================================================================
# 2. QUICK SHUFFLE
# =========================================================================
def quick_shuffle_ui():
    console.print(Panel(
        f"[bold white]QUICK SHUFFLE[/bold white] [dim]• {t('In-place random shuffle')}[/dim]",
        border_style="grey50",
        box=box.ROUNDED,
        expand=False
    ))

    input_folder = clean_path(console.input(f"\n[bold white]> 1. {t('Drag and drop dataset folder (files will be shuffled here)')}: [/bold white]"))
    if not os.path.exists(input_folder) or not os.path.isdir(input_folder):
        console.print(f"[bold white]\\[ERROR][/bold white] [dim]{t('Folder not found!')}[/dim]")
        return

    confirm = console.input(f"[bold white]> {t('Files will be renamed in the current folder. Continue? (y/n)')}: [/bold white]").strip().lower()
    if confirm not in ('y', 'yes'):
        console.print(f"[bold white]\\[SYSTEM][/bold white] [dim]{t('Cancelled.')}[/dim]")
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
        console.print(f"[bold white]\\[ERROR][/bold white] [dim]{t('No suitable images found in the folder!')}[/dim]")
        return

    num_digits = get_num_digits_choice(len(items))
    random.shuffle(items)

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
    console.print(f"\n[bold white]\\[SYSTEM][/bold white] [dim]{t('Starting process')}...[/dim]")

    progress, task = init_progress(t("Shuffling files"), len(temp_pairs))
    with progress:
        for idx, (temp_img, temp_txt, ext) in enumerate(temp_pairs, start=1):
            new_name = f"{idx:0{num_digits}d}"
            temp_img.rename(input_path / f"{new_name}{ext}")
            if temp_txt and temp_txt.exists():
                temp_txt.rename(input_path / f"{new_name}.txt")
            progress.update(task, advance=1)

    _show_summary("QUICK SHUFFLE", len(items), len(items), input_folder, time.time() - start_time)


# =========================================================================
# 3. MIRROR FLIP
# =========================================================================
def mirror_flip_ui():
    console.print(Panel(
        f"[bold white]MIRROR FLIP[/bold white] [dim]• {t('In-place horizontal flip')}[/dim]",
        border_style="grey50",
        box=box.ROUNDED,
        expand=False
    ))

    input_folder = clean_path(console.input(f"\n[bold white]> 1. {t('Drag and drop dataset folder (files will be overwritten)')}: [/bold white]"))
    if not os.path.exists(input_folder) or not os.path.isdir(input_folder):
        console.print(f"[bold white]\\[ERROR][/bold white] [dim]{t('Folder not found!')}[/dim]")
        return

    confirm = console.input(f"[bold white]> {t('WARNING! Original files will be overwritten. Continue? (y/n)')}: [/bold white]").strip().lower()
    if confirm not in ('y', 'yes'):
        console.print(f"[bold white]\\[SYSTEM][/bold white] [dim]{t('Cancelled.')}[/dim]")
        return

    input_path = Path(input_folder)
    images = [f for f in input_path.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]

    if not images:
        console.print(f"[bold white]\\[ERROR][/bold white] [dim]{t('No suitable images found in the folder!')}[/dim]")
        return

    start_time = time.time()
    console.print(f"\n[bold white]\\[SYSTEM][/bold white] [dim]{t('Starting process')}...[/dim]")

    progress, task = init_progress(t("Mirror flipping"), len(images))
    with progress:
        for img_file in images:
            try:
                with Image.open(img_file) as img:
                    img = ImageOps.mirror(img)
                    img.save(img_file)
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
                console.print(f"\n[bold white]\\[WARNING][/bold white] [dim]{img_file.name}: {e}[/dim]")
            progress.update(task, advance=1)

    _show_summary("MIRROR FLIP", len(images), len(images), input_folder, time.time() - start_time)


# =========================================================================
# 4. DATASET MULTIPLIER
# =========================================================================
def dataset_multiplier_ui():
    console.print(Panel(
        f"[bold white]DATASET MULTIPLIER[/bold white] [dim]• {t('Original + flipped -> new folder')}[/dim]",
        border_style="grey50",
        box=box.ROUNDED,
        expand=False
    ))

    input_folder = clean_path(console.input(f"\n[bold white]> 1. {t('Drag and drop source dataset folder')}: [/bold white]"))
    if not os.path.exists(input_folder) or not os.path.isdir(input_folder):
        console.print(f"[bold white]\\[ERROR][/bold white] [dim]{t('Folder not found!')}[/dim]")
        return

    output_folder = clean_path(console.input(f"[bold white]> 2. {t('Drag and drop or specify output folder')}: [/bold white]"))
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
            items.append({'img_path': img_file, 'caption': caption, 'is_flipped': False, 'ext': img_file.suffix.lower()})
            items.append({'img_path': img_file, 'caption': caption, 'is_flipped': True, 'ext': img_file.suffix.lower()})

    if not items:
        console.print(f"[bold white]\\[ERROR][/bold white] [dim]{t('No suitable images found in the folder!')}[/dim]")
        return

    num_digits = get_num_digits_choice(len(items))
    random.shuffle(items)

    start_time = time.time()
    console.print(f"\n[bold white]\\[SYSTEM][/bold white] [dim]{t('Starting process')}...[/dim]")

    progress, task = init_progress(t("Augmenting dataset"), len(items))
    with progress:
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
                console.print(f"\n[bold white]\\[WARNING][/bold white] [dim]{item['img_path'].name}: {e}[/dim]")

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

    _show_summary("DATASET MULTIPLIER", len(items)//2, len(items), output_folder, time.time() - start_time)


# =========================================================================
# 5. BATCH RENAME
# =========================================================================
def batch_rename_ui():
    console.print(Panel(
        f"[bold white]BATCH RENAME[/bold white] [dim]• {t('Clean sequential renaming')}[/dim]",
        border_style="grey50",
        box=box.ROUNDED,
        expand=False
    ))

    input_folder = clean_path(console.input(f"\n[bold white]> 1. {t('Drag and drop dataset folder')}: [/bold white]"))
    if not os.path.exists(input_folder) or not os.path.isdir(input_folder):
        console.print(f"[bold white]\\[ERROR][/bold white] [dim]{t('Folder not found!')}[/dim]")
        return

    prefix = console.input(f"[bold white]> 2. {t('Prefix (Enter = no prefix)')}: [/bold white]").strip()

    input_path = Path(input_folder)
    images = [f for f in input_path.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
    if not images:
        console.print(f"[bold white]\\[ERROR][/bold white] [dim]{t('No suitable images found in the folder!')}[/dim]")
        return

    images.sort(key=lambda x: x.name.lower())
    num_digits = get_num_digits_choice(len(images))

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
    console.print(f"\n[bold white]\\[SYSTEM][/bold white] [dim]{t('Starting process')}...[/dim]")

    progress, task = init_progress(t("Renaming files"), len(temp_files))
    with progress:
        for idx, (temp_img, temp_txt, ext) in enumerate(temp_files, start=1):
            new_name = f"{prefix}{idx:0{num_digits}d}" if prefix else f"{idx:0{num_digits}d}"
            final_img = temp_img.parent / f"{new_name}{ext}"
            temp_img.rename(final_img)
            if temp_txt and temp_txt.exists():
                temp_txt.rename(temp_txt.parent / f"{new_name}.txt")
            progress.update(task, advance=1)

    _show_summary("BATCH RENAME", len(images), len(images), input_folder, time.time() - start_time)


# =========================================================================
# MAIN MENU
# =========================================================================
def main():
    while True:
        menu_text = (
            f"[bold white][  1 ][/bold white] [white]Crop & Resize[/white]          [dim]({t('Square crop + resize, keep names')})[/dim]\n"
            f"[bold white][  2 ][/bold white] [white]Quick Shuffle[/white]          [dim]({t('In-place random shuffle')})[/dim]\n"
            f"[bold white][  3 ][/bold white] [white]Mirror Flip[/white]            [dim]({t('In-place horizontal flip, no duplicates')})[/dim]\n"
            f"[bold white][  4 ][/bold white] [white]Dataset Multiplier[/white]     [dim]({t('Original + flipped -> new folder')})[/dim]\n"
            f"[bold white][  5 ][/bold white] [white]Batch Rename[/white]           [dim]({t('Clean sequential renaming')})[/dim]\n"
            f"[dim]──────────────────────────────────────────────────────────────────────────────[/dim]\n"
            f"[bold white][  0 ][/bold white] [dim]{t('Return to main menu')}[/dim]"
        )
        console.print(Panel(
            menu_text,
            title=t("[bold white] ORAKUL STUDIO | DATASET TOOLKIT [/bold white]"),
            border_style="grey50",
            box=box.ROUNDED,
            expand=False
        ))

        choice = console.input(f"\n[bold white]> {t('Select action')} (0-5): [/bold white]").strip()

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