# ==============================================================================
#  ORAKUL CORE — AI Dataset & Model MultiTool
#  Module: LoRA SVD Distillation & Compression (CUDA Accelerated)
# ------------------------------------------------------------------------------
#  Author:      Orakul (Orakul Studio)
#  GitHub:      https://github.com/OrakulStudio
#  Civitai:     https://civitai.com/user/ORAKUL_STUDIO
#  HuggingFace: https://huggingface.co/OrakulStorm
# ------------------------------------------------------------------------------
#  Description: Rank reduction and matrix compression for LoRA weights.
#  Copyright (c) 2026 Orakul Studio. All rights reserved.
# ==============================================================================
import json
import os
import time
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import load_file, save_file
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

# =========================================================================
# УТИЛИТЫ ОБРАБОТКИ ПУТЕЙ И ВВОДА
# =========================================================================
def clean_path(path_str: str) -> str:
    """Очистка пути от кавычек и лишних пробелов при Drag-and-Drop"""
    return path_str.strip(' "\'')

def get_float_input(prompt_text: str, default_val: float) -> float:
    """Безопасный ввод вещественного числа"""
    val_str = console.input(prompt_text + f" [bold yellow][{t('По умолчанию:')} {default_val}][/bold yellow]: ").strip()
    if not val_str:
        return default_val
    try:
        return float(val_str.replace(',', '.'))
    except ValueError:
        console.print(t("[yellow]⚠ Некорректный ввод. Использовано значение по умолчанию.[/yellow]"))
        return default_val

def get_int_input(prompt_text: str, default_val: int) -> int:
    """Безопасный ввод целого числа"""
    val_str = console.input(prompt_text + f" [bold yellow][{t('По умолчанию:')} {default_val}][/bold yellow]: ").strip()
    if not val_str:
        return default_val
    try:
        return int(val_str)
    except ValueError:
        console.print(t("[yellow]⚠ Некорректный ввод. Использовано значение по умолчанию.[/yellow]"))
        return default_val

def resolve_output_path(prompt_text: str, default_filename: str = "merged_output.safetensors") -> str:
    """Умный резолвер: понимает и перетащенные папки, и имена файлов, и полные пути"""
    user_input = console.input(prompt_text)
    cleaned = clean_path(user_input)

    if not cleaned:
        cleaned = default_filename

    # Сценарий 1: Перетащили папку из Проводника
    if os.path.isdir(cleaned):
        filename = console.input(t("[bold cyan]Введи имя файла для этой папки[/bold cyan]") + f" [bold yellow][{t('По умолчанию:')} {default_filename}][/bold yellow]: ").strip()
        if not filename:
            filename = default_filename
        if not filename.endswith(".safetensors"):
            filename += ".safetensors"
        return str(Path(cleaned) / filename)

    # Сценарий 2: Написали просто имя файла (сохранит в корень Orakul_Studio_Core)
    elif not os.path.dirname(cleaned):
        filename = cleaned if cleaned else default_filename
        if not filename.endswith(".safetensors"):
            filename += ".safetensors"
        return str(Path(filename).resolve())

    # Сценарий 3: Перетащили/ввели полный путь с именем файла
    else:
        if not cleaned.endswith(".safetensors"):
            cleaned += ".safetensors"
        return cleaned


# =========================================================================
# 1. FLUX.2: BASE MODEL + LORA (CUDA Инъекция)
# =========================================================================
def merge_flux_base_lora():
    console.print(Panel(t("[bold yellow]⚡ FLUX.2 • BASE MODEL + LORA (CUDA ACCELERATED) ⚡[/bold yellow]"), expand=False))

    base_path = clean_path(console.input("\n" + t("[bold cyan]1. Перетащи базовую модель (Base .safetensors): [/bold cyan]")))
    lora_path = clean_path(console.input(t("[bold cyan]2. Перетащи LoRA Flux.2 (.safetensors): [/bold cyan]")))
    
    if not os.path.exists(base_path) or not os.path.exists(lora_path):
        console.print(t("[bold red]❌ Ошибка: Один из исходных файлов не найден![/bold red]"))
        return

    output_path = resolve_output_path("\n" + t("[bold cyan]3. Укажи имя, полный путь или перетащи папку сохранения: [/bold cyan]"), "Flux2_Base_LoRA_Merge.safetensors")
    alpha = get_float_input(t("[bold cyan]4. Коэффициент силы LoRA (Alpha)[/bold cyan]"), 1.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        console.print(t("[bold green]🚀 Подключено GPU-ускорение:[/bold green]") + f" [bold yellow]{torch.cuda.get_device_name(0)}[/bold yellow]\n")
    else:
        console.print(t("[bold yellow]⚠ CUDA не найдена, вычисления будут на CPU[/bold yellow]") + "\n")

    start_time = time.time()

    with console.status(t("[bold cyan]Загрузка базовой модели Flux.2 в RAM...[/bold cyan]"), spinner="dots"):
        base = load_file(base_path, device="cpu")

    with console.status(t("[bold yellow]Загрузка LoRA Flux.2 в RAM...[/bold yellow]"), spinner="dots"):
        lora = load_file(lora_path, device="cpu")

    matching_keys = [k for k in lora.keys() if "lora_down" in k or "lora_A" in k]
    console.print(t("[bold green]✓[/bold green] Модели загружены. Найдено LoRA-слоев: ") + f"[bold cyan]{len(matching_keys)}[/bold cyan]\n")

    processed_count = 0

    with Progress(
        SpinnerColumn("dots", style="bold yellow"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=35, style="black", complete_style="bold green", finished_style="bold green"),
        TaskProgressColumn("[bold yellow]{task.percentage:>3.0f}%"),
        TextColumn(t("• [bold magenta]{task.completed}/{task.total}[/] слоев")),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(t("Инъекция весов Flux.2"), total=len(matching_keys))

        for key in lora.keys():
            if "lora_down" in key or "lora_A" in key:
                if "lora_down" in key:
                    up_key = key.replace("lora_down", "lora_up")
                else:
                    up_key = key.replace("lora_A", "lora_B")

                if up_key in lora:
                    clean_key = key.replace(".lora_down.weight", ".weight").replace(".lora_A.weight", ".weight")
                    clean_key = clean_key.replace("lora_unet_", "").replace("diffusion_model.", "").replace("transformer.", "")

                    matched_base_key = None
                    if clean_key in base:
                        matched_base_key = clean_key
                    else:
                        for b_key in base.keys():
                            if b_key.endswith(clean_key) or clean_key.endswith(b_key):
                                matched_base_key = b_key
                                break

                    if matched_base_key:
                        w_down = lora[key].to(device, dtype=torch.float32)
                        w_up = lora[up_key].to(device, dtype=torch.float32)

                        merged_gpu = torch.matmul(w_up, w_down) * alpha

                        merged_cpu = merged_gpu.to("cpu", dtype=base[matched_base_key].dtype)
                        base[matched_base_key] += merged_cpu

                        del w_down, w_up, merged_gpu
                        processed_count += 1

                progress.update(task, advance=1)

    with console.status(t("[bold green]Сохранение монолита на диск...[/bold green]"), spinner="dots"):
        save_file(base, output_path)

    _show_summary("FLUX.2 BASE + LORA", processed_count, len(matching_keys), output_path, start_time)


# =========================================================================
# 2. FLUX.2: LORA + LORA (Конкатенация рангов)
# =========================================================================
def merge_flux_lora_lora():
    console.print(Panel(t("[bold yellow]⚡ FLUX.2 • LORA + LORA MERGER (RANK CONCAT) ⚡[/bold yellow]"), expand=False))

    lora1_path = clean_path(console.input("\n" + t("[bold cyan]1. Перетащи первую LoRA (.safetensors): [/bold cyan]")))
    weight1 = get_float_input(t("[bold cyan]   Вес первой LoRA[/bold cyan]"), 1.0)

    lora2_path = clean_path(console.input("\n" + t("[bold cyan]2. Перетащи вторую LoRA (.safetensors): [/bold cyan]")))
    weight2 = get_float_input(t("[bold cyan]   Вес второй LoRA[/bold cyan]"), 1.0)

    if not os.path.exists(lora1_path) or not os.path.exists(lora2_path):
        console.print(t("[bold red]❌ Ошибка: Файлы LoRA не найдены![/bold red]"))
        return

    output_path = resolve_output_path("\n" + t("[bold cyan]3. Укажи имя, путь или перетащи папку сохранения: [/bold cyan]"), "Flux2_LoRA_Hybrid.safetensors")

    start_time = time.time()

    metadata = {}
    try:
        with safe_open(lora1_path, framework="pt") as f:
            metadata = f.metadata() or {}
    except Exception:
        pass

    with console.status(t("[bold cyan]Загрузка LoRA 1 в RAM...[/bold cyan]"), spinner="dots"):
        l1 = load_file(lora1_path, device="cpu")

    with console.status(t("[bold yellow]Загрузка LoRA 2 в RAM...[/bold yellow]"), spinner="dots"):
        l2 = load_file(lora2_path, device="cpu")

    new_state_dict = {}
    keys_A = [k for k in l1.keys() if ".lora_A.weight" in k or ".lora_down.weight" in k]

    if not keys_A:
        console.print(t("[bold red]❌ В первой LoRA не найдено весовых матриц![/bold red]"))
        return

    console.print(t("[bold green]✓[/bold green] Модели загружены. Найдено слоев Flux.2: ") + f"[bold cyan]{len(keys_A)}[/bold cyan]\n")

    processed_count = 0
    skipped_count = 0

    with Progress(
        SpinnerColumn("dots", style="bold yellow"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=35, style="black", complete_style="bold green", finished_style="bold green"),
        TaskProgressColumn("[bold yellow]{task.percentage:>3.0f}%"),
        TextColumn(t("• [bold magenta]{task.completed}/{task.total}[/] слоев")),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(t("Сшивка гибрида Flux.2"), total=len(keys_A))

        for key_A in keys_A:
            if ".lora_A.weight" in key_A:
                key_B = key_A.replace(".lora_A.weight", ".lora_B.weight")
            else:
                key_B = key_A.replace(".lora_down.weight", ".lora_up.weight")

            if key_A in l2 and key_B in l1 and key_B in l2:
                A1, B1 = l1[key_A], l1[key_B]
                A2, B2 = l2[key_A], l2[key_B]

                B1_scaled = B1 * weight1
                B2_scaled = B2 * weight2

                new_A = torch.cat([A1, A2], dim=0)
                new_B = torch.cat([B1_scaled, B2_scaled], dim=1)

                new_state_dict[key_A] = new_A
                new_state_dict[key_B] = new_B

                alpha_key = key_A.replace(".lora_A.weight", ".alpha").replace(".lora_down.weight", ".alpha")
                if alpha_key in l1 and alpha_key in l2:
                    new_state_dict[alpha_key] = l1[alpha_key] + l2[alpha_key]
                elif alpha_key in l1:
                    new_state_dict[alpha_key] = l1[alpha_key]

                processed_count += 1
            else:
                new_state_dict[key_A] = l1[key_A]
                if key_B in l1:
                    new_state_dict[key_B] = l1[key_B]
                skipped_count += 1

            progress.update(task, advance=1)

    for k in l1.keys():
        if k not in new_state_dict and k in l2:
            new_state_dict[k] = l1[k]

    with console.status(t("[bold green]Запись гибридной LoRA Flux.2 на диск...[/bold green]"), spinner="dots"):
        save_file(new_state_dict, output_path, metadata=metadata)

    _show_summary("FLUX.2 LORA + LORA", processed_count, len(keys_A), output_path, start_time)


# =========================================================================
# 3. Z-IMAGE: LORA + LORA (Конкатенация)
# =========================================================================
def merge_z_image_lora_lora():
    console.print(Panel(t("[bold yellow]⚡ Z-IMAGE • LORA + LORA MERGER ⚡[/bold yellow]"), expand=False))

    lora1_path = clean_path(console.input("\n" + t("[bold cyan]1. Перетащи первую LoRA Z-Image (.safetensors): [/bold cyan]")))
    weight1 = get_float_input(t("[bold cyan]   Вес первой LoRA[/bold cyan]"), 1.0)

    lora2_path = clean_path(console.input("\n" + t("[bold cyan]2. Перетащи вторую LoRA Z-Image (.safetensors): [/bold cyan]")))
    weight2 = get_float_input(t("[bold cyan]   Вес второй LoRA[/bold cyan]"), 1.0)

    if not os.path.exists(lora1_path) or not os.path.exists(lora2_path):
        console.print(t("[bold red]❌ Ошибка: Файлы LoRA не найдены![/bold red]"))
        return

    output_path = resolve_output_path("\n" + t("[bold cyan]3. Укажи имя, путь или перетащи папку сохранения: [/bold cyan]"), "ZImage_LoRA_Hybrid.safetensors")

    start_time = time.time()

    with console.status(t("[bold cyan]Загрузка LoRA 1 Z-Image в RAM...[/bold cyan]"), spinner="dots"):
        l1 = load_file(lora1_path, device="cpu")

    with console.status(t("[bold yellow]Загрузка LoRA 2 Z-Image в RAM...[/bold yellow]"), spinner="dots"):
        l2 = load_file(lora2_path, device="cpu")

    new_state_dict = {}
    keys_A = [k for k in l1.keys() if "lora_A.weight" in k or "lora_down.weight" in k]

    console.print(t("[bold green]✓[/bold green] Модели загружены. Найдено слоев Z-Image: ") + f"[bold cyan]{len(keys_A)}[/bold cyan]\n")

    processed_count = 0

    with Progress(
        SpinnerColumn("dots", style="bold yellow"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=35, style="black", complete_style="bold green", finished_style="bold green"),
        TaskProgressColumn("[bold yellow]{task.percentage:>3.0f}%"),
        TextColumn(t("• [bold magenta]{task.completed}/{task.total}[/] слоев")),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(t("Конкатенация Z-Image LoRA"), total=len(keys_A))

        for key_A in keys_A:
            if "lora_A.weight" in key_A:
                key_B = key_A.replace("lora_A.weight", "lora_B.weight")
            else:
                key_B = key_A.replace("lora_down.weight", "lora_up.weight")

            if key_A in l2 and key_B in l1 and key_B in l2:
                A1, B1 = l1[key_A], l1[key_B]
                A2, B2 = l2[key_A], l2[key_B]

                B1_scaled = B1 * weight1
                B2_scaled = B2 * weight2

                new_A = torch.cat([A1, A2], dim=0)
                new_B = torch.cat([B1_scaled, B2_scaled], dim=1)

                new_state_dict[key_A] = new_A
                new_state_dict[key_B] = new_B

                alpha_key = key_A.replace(".lora_A.weight", ".alpha").replace(".lora_down.weight", ".alpha")
                if alpha_key in l1 and alpha_key in l2:
                    new_state_dict[alpha_key] = l1[alpha_key] + l2[alpha_key]

                processed_count += 1

            progress.update(task, advance=1)

    with console.status(t("[bold green]Сохранение гибридной LoRA Z-Image...[/bold green]"), spinner="dots"):
        save_file(new_state_dict, output_path)

    _show_summary("Z-IMAGE LORA + LORA", processed_count, len(keys_A), output_path, start_time)


# =========================================================================
# 4. Z-IMAGE: BASE MODEL + LORA (CUDA Инъекция)
# =========================================================================
def merge_z_image_base_lora():
    console.print(Panel(t("[bold yellow]⚡ Z-IMAGE • BASE MODEL + LORA (CUDA ACCELERATED) ⚡[/bold yellow]"), expand=False))

    base_path = clean_path(console.input("\n" + t("[bold cyan]1. Перетащи базовую модель Z-Image (.safetensors): [/bold cyan]")))
    lora_path = clean_path(console.input(t("[bold cyan]2. Перетащи LoRA Z-Image (.safetensors): [/bold cyan]")))

    if not os.path.exists(base_path) or not os.path.exists(lora_path):
        console.print(t("[bold red]❌ Ошибка: Файлы не найдены![/bold red]"))
        return

    output_path = resolve_output_path("\n" + t("[bold cyan]3. Укажи имя, путь или перетащи папку сохранения: [/bold cyan]"), "ZImage_Base_LoRA_Merge.safetensors")
    alpha = get_float_input(t("[bold cyan]4. Коэффициент альфа (сила LoRA)[/bold cyan]"), 1.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        console.print(t("[bold green]🚀 Подключено GPU-ускорение:[/bold green]") + f" [bold yellow]{torch.cuda.get_device_name(0)}[/bold yellow]\n")
    else:
        console.print(t("[bold yellow]⚠ CUDA не найдена, вычисления будут на CPU[/bold yellow]") + "\n")

    start_time = time.time()

    with console.status(t("[bold cyan]Загрузка базовой модели Z-Image в RAM...[/bold cyan]"), spinner="dots"):
        base = load_file(base_path, device="cpu")

    with console.status(t("[bold yellow]Загрузка LoRA Z-Image в RAM...[/bold yellow]"), spinner="dots"):
        lora = load_file(lora_path, device="cpu")

    keys_A = [k for k in lora.keys() if "lora_A.weight" in k or "lora_down.weight" in k]
    console.print(t("[bold green]✓[/bold green] Загружено. Найдено LoRA-слоев: ") + f"[bold cyan]{len(keys_A)}[/bold cyan]\n")

    processed_count = 0

    with Progress(
        SpinnerColumn("dots", style="bold yellow"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=35, style="black", complete_style="bold green", finished_style="bold green"),
        TaskProgressColumn("[bold yellow]{task.percentage:>3.0f}%"),
        TextColumn(t("• [bold magenta]{task.completed}/{task.total}[/] слоев")),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(t("Инъекция Z-Image"), total=len(keys_A))

        for key_A in keys_A:
            if "lora_A.weight" in key_A:
                key_B = key_A.replace("lora_A.weight", "lora_B.weight")
            else:
                key_B = key_A.replace("lora_down.weight", "lora_up.weight")

            if key_B in lora:
                clean_key = key_A.replace(".lora_A.weight", ".weight").replace(".lora_down.weight", ".weight")
                clean_key = clean_key.replace("lora_unet_", "").replace("transformer.", "").replace("diffusion_model.", "")

                matched_base_key = None
                if clean_key in base:
                    matched_base_key = clean_key
                else:
                    for b_key in base.keys():
                        if b_key.endswith(clean_key) or clean_key.endswith(b_key):
                            matched_base_key = b_key
                            break

                if matched_base_key:
                    w_A = lora[key_A].to(device, dtype=torch.float32)
                    w_B = lora[key_B].to(device, dtype=torch.float32)

                    merged_gpu = torch.matmul(w_B, w_A) * alpha

                    merged_cpu = merged_gpu.to("cpu", dtype=base[matched_base_key].dtype)
                    base[matched_base_key] += merged_cpu

                    del w_A, w_B, merged_gpu
                    processed_count += 1

            progress.update(task, advance=1)

    with console.status(t("[bold green]Сохранение монолитной модели Z-Image...[/bold green]"), spinner="dots"):
        save_file(base, output_path)

    _show_summary("Z-IMAGE BASE + LORA", processed_count, len(keys_A), output_path, start_time)


# =========================================================================
# 5. LORA SVD DISTILLATION (Сжатие гибридов до целевого ранга)
# =========================================================================
def compress_lora_svd_ui():
    console.print(Panel(t("[bold yellow]⚡ LORA SVD DISTILLATION • COMPRESSOR (CUDA ACCELERATED) ⚡[/bold yellow]"), expand=False))

    input_path = clean_path(console.input("\n" + t("[bold cyan]1. Перетащи исходную LoRA (.safetensors): [/bold cyan]")))
    if not os.path.exists(input_path):
        console.print(t("[bold red]❌ Ошибка: Исходный файл не найден![/bold red]"))
        return

    output_path = resolve_output_path("\n" + t("[bold cyan]2. Укажи имя, путь или перетащи папку сохранения: [/bold cyan]"), "Distilled_LoRA_SVD.safetensors")
    target_rank = get_int_input(t("[bold cyan]3. Введи целевой ранг (Target Rank, например: 128 или 256)[/bold cyan]"), 128)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        console.print(t("[bold green]🚀 Подключено GPU-ускорение:[/bold green]") + f" [bold yellow]{torch.cuda.get_device_name(0)}[/bold yellow]\n")
    else:
        console.print(t("[bold yellow]⚠ CUDA не найдена, вычисления будут на CPU[/bold yellow]") + "\n")

    start_time = time.time()

    metadata = {}
    try:
        with safe_open(input_path, framework="pt") as f:
            metadata = f.metadata() or {}
    except Exception:
        pass

    with console.status(t("[bold cyan]Загрузка LoRA в RAM...[/bold cyan]"), spinner="dots"):
        state_dict = load_file(input_path)

    new_state_dict = {}
    pairs = {}

    for key in state_dict.keys():
        if ".lora_down.weight" in key or ".lora_A.weight" in key:
            base = key.replace(".lora_down.weight", "").replace(".lora_A.weight", "")
            pairs.setdefault(base, {})["A"] = key
        elif ".lora_B.weight" in key or ".lora_up.weight" in key:
            base = key.replace(".lora_up.weight", "").replace(".lora_B.weight", "")
            pairs.setdefault(base, {})["B"] = key
        else:
            new_state_dict[key] = state_dict[key]

    console.print(t("[bold green]✓[/bold green] Загружено. Найдено ") + f"[bold cyan]{len(pairs)}[/bold cyan] " + t("пар матриц для SVD-дистилляции.") + "\n")

    processed_count = 0

    with Progress(
        SpinnerColumn("dots", style="bold yellow"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=35, style="black", complete_style="bold green", finished_style="bold green"),
        TaskProgressColumn("[bold yellow]{task.percentage:>3.0f}%"),
        TextColumn(t("• [bold magenta]{task.completed}/{task.total}[/] пар")),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(t("SVD Дистилляция (до Rank ") + f"{target_rank})", total=len(pairs))

        for base_key, pair in pairs.items():
            if "A" not in pair or "B" not in pair:
                progress.update(task, advance=1)
                continue

            key_A, key_B = pair["A"], pair["B"]
            w_A = state_dict[key_A].to(device, dtype=torch.float32)
            w_B = state_dict[key_B].to(device, dtype=torch.float32)

            orig_shape_A = w_A.shape

            if w_A.dim() == 4:
                w_A = w_A.squeeze(-1).squeeze(-1)
                w_B = w_B.squeeze(-1).squeeze(-1)

            # Восстанавливаем дельту весов W = B @ A
            delta_w = torch.matmul(w_B, w_A)

            # 🚀 SVD на тензорных ядрах GPU
            U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)

            r = min(target_rank, S.shape[0])
            U_r = U[:, :r]
            S_r = S[:r]
            Vh_r = Vh[:r, :]

            sqrt_S = torch.diag(torch.sqrt(S_r))
            new_B = torch.matmul(U_r, sqrt_S)
            new_A = torch.matmul(sqrt_S, Vh_r)

            if len(orig_shape_A) == 4:
                new_A = new_A.unsqueeze(-1).unsqueeze(-1)
                new_B = new_B.unsqueeze(-1).unsqueeze(-1)

            orig_dtype = state_dict[key_A].dtype
            new_state_dict[key_A] = new_A.to(orig_dtype).cpu()
            new_state_dict[key_B] = new_B.to(orig_dtype).cpu()

            alpha_key = base_key + ".alpha"
            if alpha_key in new_state_dict:
                new_state_dict[alpha_key] = torch.tensor(float(r), dtype=torch.float32)

            del w_A, w_B, delta_w, U, S, Vh, U_r, S_r, Vh_r, sqrt_S, new_A, new_B
            processed_count += 1
            progress.update(task, advance=1)

    new_metadata = dict(metadata)
    new_metadata["ss_network_dim"] = str(target_rank)
    new_metadata["ss_network_alpha"] = str(target_rank)

    with console.status(t("[bold green]Запись дистиллированной LoRA на диск...[/bold green]"), spinner="dots"):
        save_file(new_state_dict, output_path, metadata=new_metadata)

    _show_summary(f"SVD DISTILLATION (Rank {target_rank})", processed_count, len(pairs), output_path, start_time)


# =========================================================================
# УТИЛИТА ВЫВОДА СВОДКИ
# =========================================================================
def _show_summary(title: str, processed: int, total: int, output_path: str, start_time: float):
    elapsed = time.time() - start_time
    size_mb = os.path.getsize(output_path) / (1024 ** 2) if os.path.exists(output_path) else 0

    summary = Table(title=f"[bold green]{t('📊 ОТЧЕТ:')} {title}[/bold green]", show_header=True, header_style="bold cyan")
    summary.add_column(t("Параметр"), style="white")
    summary.add_column(t("Значение"), style="bold yellow")

    summary.add_row(t("Обработано слоев"), f"{processed} / {total}")
    summary.add_row(t("Сохранено по пути"), f"[bold white]{output_path}[/bold white]")
    if size_mb >= 1024:
        summary.add_row(t("Размер итогового файла"), f"[bold green]{size_mb / 1024:.2f} GB[/bold green]")
    else:
        summary.add_row(t("Размер итогового файла"), f"[bold green]{size_mb:.2f} MB[/bold green]")
    summary.add_row(t("Затрачено времени"), f"{elapsed:.2f} {t('сек')}")

    console.print("\n")
    console.print(summary)


# =========================================================================
# ГЛАВНОЕ МЕНЮ СЛИЯНИЙ
# =========================================================================
def main():
    while True:
        menu_text = (
            t("[bold cyan][1][/bold cyan] Flux.2: Base Model + LoRA (Прямая инъекция весов)") + "\n" +
            t("[bold cyan][2][/bold cyan] Flux.2: LoRA + LoRA (Конкатенация рангов r1 + r2)") + "\n" +
            t("[bold cyan][3][/bold cyan] Z-Image: LoRA + LoRA (Конкатенация рангов)") + "\n" +
            t("[bold cyan][4][/bold cyan] Z-Image: Base Model + LoRA (Инъекция с умным резолвером)") + "\n" +
            t("[bold cyan][5][/bold cyan] LoRA SVD Distillation (Сжатие гибридов до целевого ранга)") + "\n" +
            t("[bold red][0][/bold red] Назад в главное меню")
        )
        console.print(Panel(menu_text, title=t("[bold yellow]⚡ ORAKUL STUDIO • WEIGHT MERGER HUB ⚡[/bold yellow]"), border_style="yellow", expand=False))
        
        choice = console.input("\n" + t("[bold white]Выбери режим слияния (0-5): [/bold white]")).strip()

        if choice == '1':
            merge_flux_base_lora()
            break
        elif choice == '2':
            merge_flux_lora_lora()
            break
        elif choice == '3':
            merge_z_image_lora_lora()
            break
        elif choice == '4':
            merge_z_image_base_lora()
            break
        elif choice == '5':
            compress_lora_svd_ui()
            break
        elif choice == '0':
            break

if __name__ == "__main__":
    main()