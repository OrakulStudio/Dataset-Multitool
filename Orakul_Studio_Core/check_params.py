# ==============================================================================
#  ORAKUL CORE — AI Dataset & Model MultiTool
#  Module: Model Inspector
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

def inspect_model():
    console.print(Panel(
        "[bold white]ORAKUL MODEL INSPECTOR[/bold white] [dim]• Params & Metadata[/dim]",
        border_style="grey50",
        box=box.ROUNDED
    ))
    
    raw_path = console.input("\n[bold white]> Drag model (.safetensors) here: [/bold white]")
    file_path = clean_path(raw_path)

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        console.print(Panel(
            "[bold white]\\[ERROR][/bold white] [dim]File not found![/dim]",
            border_style="grey50",
            box=box.ROUNDED
        ))
        return

    start_time = time.time()
    total_params = 0
    total_tensors = 0
    metadata = {}

    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            
            for key in f.keys():
                shape = f.get_slice(key).get_shape()
                total_params += math.prod(shape)
                total_tensors += 1
    except Exception as e:
        console.print(Panel(
            f"[bold white]\\[ERROR][/bold white] [dim]Safetensors read error: {e}[/dim]",
            border_style="grey50",
            box=box.ROUNDED
        ))
        return

    elapsed = time.time() - start_time
    size_gb = os.path.getsize(file_path) / (1024 ** 3)
    billions = total_params / 1_000_000_000

    # Trigger words extraction
    trigger_words = []
    base_model = metadata.get("ss_sd_model_name", metadata.get("ss_base_model_version", "Not specified"))
    
    if "ss_trained_words" in metadata:
        try:
            words = json.loads(metadata["ss_trained_words"])
            if isinstance(words, list):
                trigger_words.extend(words)
            elif isinstance(words, str):
                trigger_words.append(words)
        except Exception:
            trigger_words.append(str(metadata["ss_trained_words"]))

    if "modelspec.trigger_phrase" in metadata:
        trigger_words.append(metadata["modelspec.trigger_phrase"])

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

    triggers_str = ", ".join(set(trigger_words)) if trigger_words else "[dim]No trigger words found in metadata[/dim]"

    table = Table(
        title=f"[bold white]MODEL INSPECTION: {os.path.basename(file_path)}[/bold white]",
        show_header=True,
        header_style="bold white",
        border_style="grey50",
        box=box.ROUNDED
    )
    table.add_column("Parameter", style="dim")
    table.add_column("Value", style="white")

    table.add_row("File", f"[white]{os.path.basename(file_path)}[/white]")
    table.add_row("Disk size", f"{size_gb:.2f} GB")
    table.add_row("Total layers / tensors", f"{total_tensors:,}")
    table.add_row("Exact parameter count", f"{total_params:,}")
    table.add_row("Parameters (billions)", f"{billions:.6f} B")
    table.add_row("Base model", f"{base_model}")
    table.add_row("Trigger words", f"{triggers_str}")
    table.add_row("Scan time", f"{elapsed:.4f} sec")

    console.print("\n")
    console.print(table)

if __name__ == "__main__":
    inspect_model()