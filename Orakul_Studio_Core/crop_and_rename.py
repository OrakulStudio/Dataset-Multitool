# ==============================================================================
#  ORAKUL CORE — AI Dataset & Model MultiTool
#  Module: Crop and Rename (TUI)
# ------------------------------------------------------------------------------
#  Author:      Orakul (Orakul Studio)
#  GitHub:      https://github.com/OrakulStudio
#  Civitai:     https://civitai.com/user/ORAKUL_STUDIO
#  HuggingFace: https://huggingface.co/OrakulStorm
# ------------------------------------------------------------------------------
#  Description: Smart square crop + resize + optional sequential renaming.
#               Full TUI version with adaptive layout.
#  Copyright (c) 2026 Orakul Studio. All rights reserved.
# ==============================================================================

import os
import re
import sys
import shutil
import time
from pathlib import Path
from PIL import Image

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import box
from rich.align import Align

console = Console()

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', str(s))]

def make_header(title: str) -> Panel:
    return Panel(
        Align.center(f"[bold white]{title}[/bold white]"),
        border_style="grey50",
        box=box.ROUNDED,
        padding=(0, 1)
    )

def make_info_panel(files_count: int, size: int, prefix: str) -> Panel:
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column(style="white", width=18)
    table.add_column(style="dim")
    table.add_row("Files to process:", str(files_count))
    table.add_row("Target size:", f"{size}x{size}")
    table.add_row("Prefix:", prefix if prefix else "Keep original names")
    return Panel(table, border_style="grey50", box=box.ROUNDED, title="[bold white] SETTINGS [/bold white]", title_align="left")

def process_dataset_tui(input_dir: Path, output_dir: Path, size: int = 1024, prefix: str = ""):
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

    if not input_dir.exists():
        console.print(Panel("[bold white]\\[ERROR][/bold white] [dim]Source folder not found[/dim]", border_style="grey50", box=box.ROUNDED))
        return

    files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    files.sort(key=natural_sort_key)

    if not files:
        console.print(Panel("[dim]No suitable images found[/dim]", border_style="grey50", box=box.ROUNDED))
        return

    # Ask about txt sync
    matched_txt = sum(1 for f in files if f.with_suffix('.txt').exists())
    sync_txt = False
    if matched_txt > 0:
        console.print(Panel(f"[white]Found {matched_txt} paired .txt files[/white]", border_style="grey50", box=box.ROUNDED))
        choice = console.input("[bold white]> Transfer and rename them together? [Y/n]: [/bold white]").strip().lower()
        if choice in ('y', 'yes', ''):
            sync_txt = True

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare progress
    progress = Progress(
        TextColumn("[white]{task.description}[/white]"),
        BarColumn(bar_width=None, style="grey50", complete_style="white"),
        TextColumn("[white]{task.percentage:>3.0f}%[/white]"),
        TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
        TimeElapsedColumn(),
        console=console,
        expand=True
    )
    task_id = progress.add_task("Processing...", total=len(files))

    success_count = 0
    log_lines = []

    def generate_ui():
        header = make_header("ORAKUL STUDIO — CROP & RENAME")
        info = make_info_panel(len(files), size, prefix)
        prog_panel = Panel(progress, border_style="grey50", box=box.ROUNDED, title="[bold white] PROGRESS [/bold white]", title_align="left")
        
        # Show last few log lines
        log_text = "\n".join(log_lines[-8:]) if log_lines else "[dim]Waiting...[/dim]"
        log_panel = Panel(log_text, border_style="grey50", box=box.ROUNDED, title="[bold white] LOG [/bold white]", title_align="left")
        
        return Group(header, info, prog_panel, log_panel)

    with Live(generate_ui(), refresh_per_second=12, console=console) as live:
        for i, file_path in enumerate(files, start=1):
            ext = file_path.suffix.lower()
            original_stem = file_path.stem

            if prefix:
                new_stem = f"{prefix}_{i:05d}"
            else:
                new_stem = original_stem

            new_filename = f"{new_stem}{ext}"
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

                if sync_txt:
                    src_txt = file_path.with_suffix('.txt')
                    if src_txt.exists():
                        shutil.copy2(src_txt, output_dir / f"{new_stem}.txt")

                success_count += 1
                log_lines.append(f"[white]• {file_path.name}[/white] → [dim]{new_filename}[/dim]")
            except Exception as e:
                log_lines.append(f"[dim]• {file_path.name} → Error: {e}[/dim]")

            progress.update(task_id, advance=1)
            live.update(generate_ui())

    # Final screen
    console.print()
    console.print(Panel(
        f"[white]Successfully processed:[/white] [dim]{success_count}/{len(files)}[/dim]\n"
        f"[white]Saved to:[/white] [dim]{output_dir}[/dim]",
        title="[bold white] DONE [/bold white]",
        border_style="grey50",
        box=box.ROUNDED
    ))

def main():
    if len(sys.argv) > 1:
        base_folder = Path(sys.argv[1]).resolve()
    else:
        base_folder = Path(console.input("[bold white]> Drag and drop the dataset folder here: [/bold white]").strip(' "\'')).resolve()

    if not base_folder.exists():
        console.print(Panel("[bold white]\\[ERROR][/bold white] [dim]Path does not exist[/dim]", border_style="grey50", box=box.ROUNDED))
        return

    input_dir = base_folder / "raw" if (base_folder / "raw").exists() else base_folder
    output_dir = base_folder / "ready"

    size_input = console.input("[bold white]> Square size in pixels [Enter = 1024]: [/bold white]").strip()
    size = int(size_input) if size_input.isdigit() else 1024

    prefix = console.input("[bold white]> Prefix for files [Enter = keep original names]: [/bold white]").strip()

    process_dataset_tui(input_dir, output_dir, size, prefix)

if __name__ == "__main__":
    main()