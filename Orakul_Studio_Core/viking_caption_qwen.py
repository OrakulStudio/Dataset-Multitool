# ==============================================================================
#  ORAKUL CORE — AI Dataset & Model MultiTool
#  Module: Viking Caption (Qwen-VL Auto-Captioner)
# ------------------------------------------------------------------------------
#  Author:      Orakul (Orakul Studio)
#  GitHub:      https://github.com/OrakulStudio
#  Civitai:     https://civitai.com/user/ORAKUL_STUDIO
#  HuggingFace: https://huggingface.co/OrakulStorm
# ------------------------------------------------------------------------------
#  Description: High-performance vision-language auto-captioning system.
#  Copyright (c) 2026 Orakul Studio. All rights reserved.
# ==============================================================================

import os
import sys
import json
import logging
from pathlib import Path

logging.getLogger("transformers").setLevel(logging.ERROR)

# === DYNAMIC CACHE ROUTING ===
config_file_cache = Path(__file__).parent / "paths.json"
cache_path = None

if config_file_cache.exists():
    try:
        with open(config_file_cache, "r", encoding="utf-8") as f:
            saved_path = json.load(f).get("hf_cache_dir", r"xxx:\AI_Models\Cache")
            drive = Path(saved_path).drive

            if drive and Path(f"{drive}/").exists():
                cache_path = saved_path
    except Exception:
        pass

if not cache_path:
    cache_path = r"xxx:\AI_Models\Cache"
    if not Path("xxx:/").exists():
        user_drive = input("Specify your drive letter for downloading Qwen (e.g., C, E, G): ").strip().upper()
        if user_drive:
            cache_path = rf"{user_drive}:\AI_Models\Cache"

    with open(config_file_cache, "w", encoding="utf-8") as f:
        current_config = {}
        if config_file_cache.exists():
            try:
                with open(config_file_cache, "r", encoding="utf-8") as fr:
                    current_config = json.load(fr)
            except Exception:
                pass
        current_config["hf_cache_dir"] = cache_path
        json.dump(current_config, f, indent=4)
        
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HOME"] = cache_path

# --- KERNELS PATCH FOR STABLE PYTORCH ---
import glob
import shutil

kernels_dir = os.path.join(cache_path, "hub", "kernels--kernels-community--finegrained-fp8")
if os.path.exists(kernels_dir):
    for file_path in glob.glob(os.path.join(kernels_dir, "**", "*.py"), recursive=True):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            modified = False
            if "block_size: list[int]" in content:
                content = content.replace("block_size: list[int]", "block_size: List[int]")
                modified = True
            if "block_size: list[int] | None" in content:
                content = content.replace("block_size: list[int] | None", "block_size: List[int] | None")
                modified = True
                
            if modified:
                if "from typing import List" not in content:
                    content = "from typing import List\n" + content
                    
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                    
                pycache_dir = os.path.join(os.path.dirname(file_path), "__pycache__")
                if os.path.exists(pycache_dir):
                    shutil.rmtree(pycache_dir, ignore_errors=True)
        except Exception:
            pass

import torch

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", getattr(torch, "float8_e4m3fn", torch.uint8))

from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from rich.console import Console
from rich.panel import Panel
from rich import box
from rich.align import Align
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TimeElapsedColumn
)

console = Console()

def draw_header(status_text=""):
    os.system("cls" if os.name == 'nt' else 'clear')
    header_str = "[bold white]ORAKUL STUDIO — VIKING CAPTION (QWEN-VL)[/bold white]"
    if status_text:
        header_str += f" [dim]| {status_text}[/dim]"
    console.print(Panel(Align.center(header_str), border_style="grey50", box=box.ROUNDED))
    console.print()

def main():
    draw_header("INITIALIZING")

    if len(sys.argv) < 4:
        console.print(Panel(
            "[bold white]\\[ERROR][/bold white] [dim]Script must be run via run.py (model ID missing)![/dim]",
            border_style="grey50",
            box=box.ROUNDED
        ))
        sys.exit(1)

    TARGET_FOLDER = sys.argv[1] 
    PROMPT_FILE = sys.argv[2]
    MODEL_NAME = sys.argv[3]

    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            SYSTEM_PROMPT = f.read().strip()
    except Exception as e:
        console.print(Panel(
            f"[bold white]\\[ERROR][/bold white] [dim]Prompt reading error: {e}[/dim]",
            border_style="grey50",
            box=box.ROUNDED
        ))
        sys.exit(1)

    console.print(Panel(
        f"[bold white]\\[SYSTEM][/bold white] [dim]Loading model: {MODEL_NAME}[/dim]",
        border_style="grey50",
        box=box.ROUNDED
    ))
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
        trust_remote_code=True
    )
    
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.PNG', '.JPG', '.JPEG')
    image_paths = [p for p in Path(TARGET_FOLDER).glob("*") if p.suffix in valid_exts]
    
    if not image_paths:
        console.print(Panel(
            f"[bold white]\\[ERROR][/bold white] [dim]No images found in folder: {TARGET_FOLDER}[/dim]",
            border_style="grey50",
            box=box.ROUNDED
        ))
        return
        
    draw_header("ANALYZING DATASET")
    console.print(Panel(
        f"[white]Target Folder:[/white] [dim]{TARGET_FOLDER}[/dim]\n"
        f"[white]Images found:[/white] [dim]{len(image_paths)}[/dim]",
        title="[bold white] TASK INFO [/bold white]",
        border_style="grey50",
        box=box.ROUNDED
    ))
    console.print()

    progress = Progress(
        SpinnerColumn(style="white"),
        TextColumn("[white]{task.description}[/white]"),
        BarColumn(complete_style="white", finished_style="bold white", pulse_style="dim"),
        TextColumn("[white]{task.percentage:>3.0f}%[/white]"),
        TextColumn("[white]{task.completed}/{task.total}[/white]"),
        TextColumn("[dim]Time:[/dim]"),
        TimeElapsedColumn(),
        console=console,
        expand=False
    )

    with progress:
        task = progress.add_task("Processing...", total=len(image_paths))
        
        prog_panel = Panel(
            progress,
            border_style="grey50",
            box=box.ROUNDED,
            title="[bold white] PROGRESS [/bold white]"
        )
        console.print(prog_panel)

        for img_path in image_paths:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(img_path.resolve())},
                            {"type": "text", "text": SYSTEM_PROMPT}
                        ]
                    }
                ]
                
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to("cuda")
                
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                caption = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                output_file = Path(TARGET_FOLDER) / f"{img_path.stem}.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(caption.strip())
                    
                progress.advance(task)
                    
            except Exception:
                pass
                
    console.print()
    console.print(Panel(
        Align.center("[bold white]PROCESS COMPLETED. VRAM FREED.[/bold white]"),
        border_style="grey50",
        box=box.ROUNDED
    ))

if __name__ == "__main__":
    main()