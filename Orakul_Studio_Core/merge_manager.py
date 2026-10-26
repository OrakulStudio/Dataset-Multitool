# ==============================================================================
#  ORAKUL CORE — AI Dataset & Model MultiTool
#  Module: Weight Merger & SVD Lab
# ------------------------------------------------------------------------------
#  Author:      Orakul (Orakul Studio)
#  GitHub:      https://github.com/OrakulStudio
#  Civitai:     https://civitai.com/user/ORAKUL_STUDIO
#  HuggingFace: https://huggingface.co/OrakulStorm
# ------------------------------------------------------------------------------
#  Description: Rank reduction, matrix compression and LoRA merging tools.
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

def clean_path(path_str: str) -> str:
    return path_str.strip(' "\'')

def get_float_input(prompt_text: str, default_val: float) -> float:
    val_str = console.input(f"{prompt_text} [dim](Default: {default_val})[/dim]: ").strip()
    if not val_str:
        return default_val
    try:
        return float(val_str.replace(',', '.'))
    except ValueError:
        console.print("[dim]Invalid input. Using default value.[/dim]")
        return default_val

def get_int_input(prompt_text: str, default_val: int) -> int:
    val_str = console.input(f"{prompt_text} [dim](Default: {default_val})[/dim]: ").strip()
    if not val_str:
        return default_val
    try:
        return int(val_str)
    except ValueError:
        console.print("[dim]Invalid input. Using default value.[/dim]")
        return default_val

def resolve_output_path(prompt_text: str, default_filename: str = "merged_output.safetensors") -> str:
    user_input = console.input(prompt_text)
    cleaned = clean_path(user_input)

    if not cleaned:
        cleaned = default_filename

    if os.path.isdir(cleaned):
        filename = console.input(f"[bold white]> Filename for this folder [dim](Default: {default_filename})[/dim]: [/bold white]").strip()
        if not filename:
            filename = default_filename
        if not filename.endswith(".safetensors"):
            filename += ".safetensors"
        return str(Path(cleaned) / filename)

    elif not os.path.dirname(cleaned):
        filename = cleaned if cleaned else default_filename
        if not filename.endswith(".safetensors"):
            filename += ".safetensors"
        return str(Path(filename).resolve())

    else:
        if not cleaned.endswith(".safetensors"):
            cleaned += ".safetensors"
        return cleaned

def _show_summary(title: str, processed: int, total: int, output_path: str, start_time: float):
    elapsed = time.time() - start_time
    size_mb = os.path.getsize(output_path) / (1024 ** 2) if os.path.exists(output_path) else 0

    summary = Table(
        title=f"[bold white]REPORT: {title}[/bold white]",
        show_header=True,
        header_style="bold white",
        border_style="grey50",
        box=box.ROUNDED
    )
    summary.add_column("Parameter", style="dim")
    summary.add_column("Value", style="white")

    summary.add_row("Layers processed", f"{processed} / {total}")
    summary.add_row("Saved to", f"[white]{output_path}[/white]")
    if size_mb >= 1024:
        summary.add_row("File size", f"{size_mb / 1024:.2f} GB")
    else:
        summary.add_row("File size", f"{size_mb:.2f} MB")
    summary.add_row("Time elapsed", f"{elapsed:.2f} sec")

    console.print("\n")
    console.print(summary)

# =========================================================================
# 1. FLUX.2: BASE MODEL + LORA
# =========================================================================
def merge_flux_base_lora():
    console.print(Panel(
        "[bold white]FLUX.2 • BASE MODEL + LORA[/bold white] [dim](CUDA Accelerated)[/dim]",
        border_style="grey50",
        box=box.ROUNDED
    ))

    base_path = clean_path(console.input("\n[bold white]> 1. Drag base model (.safetensors): [/bold white]"))
    lora_path = clean_path(console.input("[bold white]> 2. Drag Flux.2 LoRA (.safetensors): [/bold white]"))
    
    if not os.path.exists(base_path) or not os.path.exists(lora_path):
        console.print(Panel("[bold white]\\[ERROR][/bold white] [dim]One of the source files not found![/dim]", border_style="grey50", box=box.ROUNDED))
        return

    output_path = resolve_output_path("\n[bold white]> 3. Output name / path / folder: [/bold white]", "Flux2_Base_LoRA_Merge.safetensors")
    alpha = get_float_input("[bold white]> 4. LoRA strength (Alpha)[/bold white]", 1.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        console.print(Panel(f"[white]GPU acceleration:[/white] [dim]{torch.cuda.get_device_name(0)}[/dim]", border_style="grey50", box=box.ROUNDED))
    else:
        console.print(Panel("[dim]CUDA not found, using CPU[/dim]", border_style="grey50", box=box.ROUNDED))

    start_time = time.time()

    with console.status("[dim]Loading base model into RAM...[/dim]", spinner="dots"):
        base = load_file(base_path, device="cpu")

    with console.status("[dim]Loading LoRA into RAM...[/dim]", spinner="dots"):
        lora = load_file(lora_path, device="cpu")

    matching_keys = [k for k in lora.keys() if "lora_down" in k or "lora_A" in k]
    console.print(Panel(f"[white]Models loaded. LoRA layers found:[/white] [dim]{len(matching_keys)}[/dim]", border_style="grey50", box=box.ROUNDED))

    processed_count = 0

    with Progress(
        SpinnerColumn(style="white"),
        TextColumn("[white]{task.description}[/white]"),
        BarColumn(bar_width=35, style="grey50", complete_style="white", finished_style="bold white"),
        TaskProgressColumn("[white]{task.percentage:>3.0f}%[/white]"),
        TextColumn("[dim]{task.completed}/{task.total} layers[/dim]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Injecting Flux.2 weights", total=len(matching_keys))

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

    with console.status("[dim]Saving merged model to disk...[/dim]", spinner="dots"):
        save_file(base, output_path)

    _show_summary("FLUX.2 BASE + LORA", processed_count, len(matching_keys), output_path, start_time)

# =========================================================================
# 2. FLUX.2: LORA + LORA
# =========================================================================
def merge_flux_lora_lora():
    console.print(Panel(
        "[bold white]FLUX.2 • LORA + LORA MERGER[/bold white] [dim](Rank Concat)[/dim]",
        border_style="grey50",
        box=box.ROUNDED
    ))

    lora1_path = clean_path(console.input("\n[bold white]> 1. Drag first LoRA (.safetensors): [/bold white]"))
    weight1 = get_float_input("[bold white]   Weight of first LoRA[/bold white]", 1.0)

    lora2_path = clean_path(console.input("\n[bold white]> 2. Drag second LoRA (.safetensors): [/bold white]"))
    weight2 = get_float_input("[bold white]   Weight of second LoRA[/bold white]", 1.0)

    if not os.path.exists(lora1_path) or not os.path.exists(lora2_path):
        console.print(Panel("[bold white]\\[ERROR][/bold white] [dim]LoRA files not found![/dim]", border_style="grey50", box=box.ROUNDED))
        return

    output_path = resolve_output_path("\n[bold white]> 3. Output name / path / folder: [/bold white]", "Flux2_LoRA_Hybrid.safetensors")

    start_time = time.time()

    metadata = {}
    try:
        with safe_open(lora1_path, framework="pt") as f:
            metadata = f.metadata() or {}
    except Exception:
        pass

    with console.status("[dim]Loading LoRA 1 into RAM...[/dim]", spinner="dots"):
        l1 = load_file(lora1_path, device="cpu")

    with console.status("[dim]Loading LoRA 2 into RAM...[/dim]", spinner="dots"):
        l2 = load_file(lora2_path, device="cpu")

    new_state_dict = {}
    keys_A = [k for k in l1.keys() if ".lora_A.weight" in k or ".lora_down.weight" in k]

    if not keys_A:
        console.print(Panel("[bold white]\\[ERROR][/bold white] [dim]No weight matrices found in first LoRA![/dim]", border_style="grey50", box=box.ROUNDED))
        return

    console.print(Panel(f"[white]Models loaded. Flux.2 layers found:[/white] [dim]{len(keys_A)}[/dim]", border_style="grey50", box=box.ROUNDED))

    processed_count = 0

    with Progress(
        SpinnerColumn(style="white"),
        TextColumn("[white]{task.description}[/white]"),
        BarColumn(bar_width=35, style="grey50", complete_style="white", finished_style="bold white"),
        TaskProgressColumn("[white]{task.percentage:>3.0f}%[/white]"),
        TextColumn("[dim]{task.completed}/{task.total} layers[/dim]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Merging Flux.2 hybrid", total=len(keys_A))

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

            progress.update(task, advance=1)

    for k in l1.keys():
        if k not in new_state_dict and k in l2:
            new_state_dict[k] = l1[k]

    with console.status("[dim]Writing hybrid LoRA to disk...[/dim]", spinner="dots"):
        save_file(new_state_dict, output_path, metadata=metadata)

    _show_summary("FLUX.2 LORA + LORA", processed_count, len(keys_A), output_path, start_time)

# =========================================================================
# 3. Z-IMAGE: LORA + LORA
# =========================================================================
def merge_z_image_lora_lora():
    console.print(Panel(
        "[bold white]Z-IMAGE • LORA + LORA MERGER[/bold white]",
        border_style="grey50",
        box=box.ROUNDED
    ))

    lora1_path = clean_path(console.input("\n[bold white]> 1. Drag first Z-Image LoRA (.safetensors): [/bold white]"))
    weight1 = get_float_input("[bold white]   Weight of first LoRA[/bold white]", 1.0)

    lora2_path = clean_path(console.input("\n[bold white]> 2. Drag second Z-Image LoRA (.safetensors): [/bold white]"))
    weight2 = get_float_input("[bold white]   Weight of second LoRA[/bold white]", 1.0)

    if not os.path.exists(lora1_path) or not os.path.exists(lora2_path):
        console.print(Panel("[bold white]\\[ERROR][/bold white] [dim]LoRA files not found![/dim]", border_style="grey50", box=box.ROUNDED))
        return

    output_path = resolve_output_path("\n[bold white]> 3. Output name / path / folder: [/bold white]", "ZImage_LoRA_Hybrid.safetensors")

    start_time = time.time()

    with console.status("[dim]Loading LoRA 1 into RAM...[/dim]", spinner="dots"):
        l1 = load_file(lora1_path, device="cpu")

    with console.status("[dim]Loading LoRA 2 into RAM...[/dim]", spinner="dots"):
        l2 = load_file(lora2_path, device="cpu")

    new_state_dict = {}
    keys_A = [k for k in l1.keys() if "lora_A.weight" in k or "lora_down.weight" in k]

    console.print(Panel(f"[white]Models loaded. Z-Image layers found:[/white] [dim]{len(keys_A)}[/dim]", border_style="grey50", box=box.ROUNDED))

    processed_count = 0

    with Progress(
        SpinnerColumn(style="white"),
        TextColumn("[white]{task.description}[/white]"),
        BarColumn(bar_width=35, style="grey50", complete_style="white", finished_style="bold white"),
        TaskProgressColumn("[white]{task.percentage:>3.0f}%[/white]"),
        TextColumn("[dim]{task.completed}/{task.total} layers[/dim]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Concatenating Z-Image LoRA", total=len(keys_A))

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

    with console.status("[dim]Saving hybrid Z-Image LoRA...[/dim]", spinner="dots"):
        save_file(new_state_dict, output_path)

    _show_summary("Z-IMAGE LORA + LORA", processed_count, len(keys_A), output_path, start_time)

# =========================================================================
# 4. Z-IMAGE: BASE MODEL + LORA
# =========================================================================
def merge_z_image_base_lora():
    console.print(Panel(
        "[bold white]Z-IMAGE • BASE MODEL + LORA[/bold white] [dim](CUDA Accelerated)[/dim]",
        border_style="grey50",
        box=box.ROUNDED
    ))

    base_path = clean_path(console.input("\n[bold white]> 1. Drag Z-Image base model (.safetensors): [/bold white]"))
    lora_path = clean_path(console.input("[bold white]> 2. Drag Z-Image LoRA (.safetensors): [/bold white]"))

    if not os.path.exists(base_path) or not os.path.exists(lora_path):
        console.print(Panel("[bold white]\\[ERROR][/bold white] [dim]Files not found![/dim]", border_style="grey50", box=box.ROUNDED))
        return

    output_path = resolve_output_path("\n[bold white]> 3. Output name / path / folder: [/bold white]", "ZImage_Base_LoRA_Merge.safetensors")
    alpha = get_float_input("[bold white]> 4. LoRA strength (Alpha)[/bold white]", 1.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        console.print(Panel(f"[white]GPU acceleration:[/white] [dim]{torch.cuda.get_device_name(0)}[/dim]", border_style="grey50", box=box.ROUNDED))
    else:
        console.print(Panel("[dim]CUDA not found, using CPU[/dim]", border_style="grey50", box=box.ROUNDED))

    start_time = time.time()

    with console.status("[dim]Loading base model into RAM...[/dim]", spinner="dots"):
        base = load_file(base_path, device="cpu")

    with console.status("[dim]Loading LoRA into RAM...[/dim]", spinner="dots"):
        lora = load_file(lora_path, device="cpu")

    keys_A = [k for k in lora.keys() if "lora_A.weight" in k or "lora_down.weight" in k]
    console.print(Panel(f"[white]Loaded. LoRA layers found:[/white] [dim]{len(keys_A)}[/dim]", border_style="grey50", box=box.ROUNDED))

    processed_count = 0

    with Progress(
        SpinnerColumn(style="white"),
        TextColumn("[white]{task.description}[/white]"),
        BarColumn(bar_width=35, style="grey50", complete_style="white", finished_style="bold white"),
        TaskProgressColumn("[white]{task.percentage:>3.0f}%[/white]"),
        TextColumn("[dim]{task.completed}/{task.total} layers[/dim]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Injecting Z-Image weights", total=len(keys_A))

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

    with console.status("[dim]Saving monolithic Z-Image model...[/dim]", spinner="dots"):
        save_file(base, output_path)

    _show_summary("Z-IMAGE BASE + LORA", processed_count, len(keys_A), output_path, start_time)

# =========================================================================
# 5. LORA SVD DISTILLATION
# =========================================================================
def compress_lora_svd_ui():
    console.print(Panel(
        "[bold white]LORA SVD DISTILLATION • COMPRESSOR[/bold white] [dim](CUDA Accelerated)[/dim]",
        border_style="grey50",
        box=box.ROUNDED
    ))

    input_path = clean_path(console.input("\n[bold white]> 1. Drag source LoRA (.safetensors): [/bold white]"))
    if not os.path.exists(input_path):
        console.print(Panel("[bold white]\\[ERROR][/bold white] [dim]Source file not found![/dim]", border_style="grey50", box=box.ROUNDED))
        return

    output_path = resolve_output_path("\n[bold white]> 2. Output name / path / folder: [/bold white]", "Distilled_LoRA_SVD.safetensors")
    target_rank = get_int_input("[bold white]> 3. Target Rank (e.g. 128 or 256)[/bold white]", 128)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        console.print(Panel(f"[white]GPU acceleration:[/white] [dim]{torch.cuda.get_device_name(0)}[/dim]", border_style="grey50", box=box.ROUNDED))
    else:
        console.print(Panel("[dim]CUDA not found, using CPU[/dim]", border_style="grey50", box=box.ROUNDED))

    start_time = time.time()

    metadata = {}
    try:
        with safe_open(input_path, framework="pt") as f:
            metadata = f.metadata() or {}
    except Exception:
        pass

    with console.status("[dim]Loading LoRA into RAM...[/dim]", spinner="dots"):
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

    console.print(Panel(f"[white]Loaded. Matrix pairs for SVD:[/white] [dim]{len(pairs)}[/dim]", border_style="grey50", box=box.ROUNDED))

    processed_count = 0

    with Progress(
        SpinnerColumn(style="white"),
        TextColumn("[white]{task.description}[/white]"),
        BarColumn(bar_width=35, style="grey50", complete_style="white", finished_style="bold white"),
        TaskProgressColumn("[white]{task.percentage:>3.0f}%[/white]"),
        TextColumn("[dim]{task.completed}/{task.total} pairs[/dim]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"SVD Distillation (to Rank {target_rank})", total=len(pairs))

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

            delta_w = torch.matmul(w_B, w_A)

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

    with console.status("[dim]Writing distilled LoRA to disk...[/dim]", spinner="dots"):
        save_file(new_state_dict, output_path, metadata=new_metadata)

    _show_summary(f"SVD DISTILLATION (Rank {target_rank})", processed_count, len(pairs), output_path, start_time)

# =========================================================================
# 6. Z-IMAGE SAEP-SVD DISTILLATION
# =========================================================================
def compress_z_image_saep_svd():
    console.print(Panel(
        "[bold white]Z-IMAGE SAEP-SVD DISTILLATION[/bold white]\n"
        "[dim]Sign Alignment & Energy Preservation — Orakul Studio Custom[/dim]",
        border_style="grey50",
        box=box.ROUNDED
    ))

    input_path = clean_path(console.input("\n[bold white]> 1. Drag high-rank Z-Image LoRA (.safetensors): [/bold white]"))
    if not os.path.exists(input_path):
        console.print(Panel("[bold white]\\[ERROR][/bold white] [dim]Source file not found![/dim]", border_style="grey50", box=box.ROUNDED))
        return

    output_path = resolve_output_path("\n[bold white]> 2. Output name / path / folder: [/bold white]", "ZImage_SAEP_Distilled.safetensors")
    target_rank = get_int_input("[bold white]> 3. Target Rank (e.g. 256)[/bold white]", 256)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        console.print(Panel(f"[white]GPU acceleration:[/white] [dim]{torch.cuda.get_device_name(0)}[/dim]", border_style="grey50", box=box.ROUNDED))

    start_time = time.time()
    metadata = {}
    try:
        with safe_open(input_path, framework="pt") as f:
            metadata = f.metadata() or {}
    except Exception:
        pass

    with console.status("[dim]Loading Z-Image LoRA into RAM...[/dim]", spinner="dots"):
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

    processed_count = 0

    with Progress(
        SpinnerColumn(style="white"),
        TextColumn("[white]{task.description}[/white]"),
        BarColumn(bar_width=35, style="grey50", complete_style="white", finished_style="bold white"),
        TaskProgressColumn("[white]{task.percentage:>3.0f}%[/white]"),
        TextColumn("[dim]{task.completed}/{task.total} matrices[/dim]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"SAEP-SVD Distillation (to Rank {target_rank})", total=len(pairs))

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

            delta_w = torch.matmul(w_B, w_A)
            
            U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
            
            r = min(target_rank, S.shape[0])
            
            # Sign Alignment
            u_sums = torch.sum(U[:, :r], dim=0)
            sign_mask = torch.where(u_sums < 0, torch.tensor(-1.0, device=device), torch.tensor(1.0, device=device))
            
            U_r = U[:, :r] * sign_mask
            Vh_r = Vh[:r, :] * sign_mask.unsqueeze(1)
            
            # Energy Preservation
            S_r = S[:r].clone()
            if r < S.shape[0]:
                residual_energy = torch.sqrt(torch.sum(S[r:] ** 2))
                inject_start = max(0, r - int(r * 0.1))
                injection_per_value = residual_energy / (r - inject_start)
                S_r[inject_start:] += injection_per_value

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
    new_metadata["distillation_method"] = "SAEP-SVD (Orakul Studio)"

    with console.status("[dim]Writing SAEP-distilled LoRA to disk...[/dim]", spinner="dots"):
        save_file(new_state_dict, output_path, metadata=new_metadata)

    _show_summary(f"SAEP-SVD (Rank {target_rank})", processed_count, len(pairs), output_path, start_time)

# =========================================================================
# MAIN MENU
# =========================================================================
def main():
    while True:
        menu_text = (
            "[bold white][ 1 ][/bold white] Flux.2: Base Model + LoRA (Direct weight injection)\n"
            "[bold white][ 2 ][/bold white] Flux.2: LoRA + LoRA (Rank concatenation r1 + r2)\n"
            "[bold white][ 3 ][/bold white] Z-Image: LoRA + LoRA (Rank concatenation)\n"
            "[bold white][ 4 ][/bold white] Z-Image: Base Model + LoRA (Smart resolver injection)\n"
            "[bold white][ 5 ][/bold white] LoRA SVD Distillation (Standard SVD compression)\n"
            "[bold white][ 6 ][/bold white] Z-Image: SAEP-SVD Distillation (Orakul Studio Custom)\n"
            "[bold white][ 0 ][/bold white] Return to main menu"
        )
        console.print(Panel(
            menu_text,
            title="[bold white] WEIGHT MERGER HUB [/bold white]",
            border_style="grey50",
            box=box.ROUNDED
        ))
        
        choice = console.input("\n[bold white]> Select merge mode (0-6): [/bold white]").strip()

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
        elif choice == '6':
            compress_z_image_saep_svd()
            break
        elif choice == '0':
            break

if __name__ == "__main__":
    main()
