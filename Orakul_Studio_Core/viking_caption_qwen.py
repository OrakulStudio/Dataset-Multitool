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
from pathlib import Path

# --- ФУНКЦИЯ ПЕРЕВОДА ИЗ ГЛАВНОГО ФАЙЛА ---
def t(text: str) -> str:
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

# === ДИНАМИЧЕСКОЕ ПЕРЕНАПРАВЛЕНИЕ КЭША (PORTABLE-READY) ===
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
        user_drive = input(t("Укажите свою букву диска для скачивания Qwen (например, C, E, G): ")).strip().upper()
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
# =========================================================


# --- АВТОМАТИЧЕСКИЙ ПАТЧ ВСЕХ ФАЙЛОВ KERNELS ДЛЯ СТАБИЛЬНОГО PYTORCH ---
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
# ----------------------------------------------------------------------


import torch

# --- ПАТЧ ДЛЯ QWEN 30B FP8 НА СТАБИЛЬНОМ PYTORCH ---
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", getattr(torch, "float8_e4m3fn", torch.uint8))
# ---------------------------------------------------

from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from rich.console import Console
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TimeElapsedColumn, 
    TimeRemainingColumn
)

# Инициализация красивой консоли
console = Console()

def main():
    # 1. Проверяем и забираем аргументы из пульта (включая модель из sys.argv[3])
    if len(sys.argv) < 4:
        console.print(t("[bold red][ERROR] Скрипт нужно запускать через пульт run.py (не передан ID модели)![/bold red]"))
        sys.exit(1)

    TARGET_FOLDER = sys.argv[1] 
    PROMPT_FILE = sys.argv[2]
    MODEL_NAME = sys.argv[3]

    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            SYSTEM_PROMPT = f.read().strip()
    except Exception as e:
        console.print(t("[bold red][ERROR] Ошибка чтения промпта: ") + f"{e}[/bold red]")
        sys.exit(1)

    console.print(t("[bold cyan][ORAKUL SYSTEM][/bold cyan] [bold yellow]Инициализация интеллектуального монстра: ") + MODEL_NAME + t("...[/bold yellow]"))
    
    # 2. Загружаем процессор
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # 3. Инициализируем модель (чего и не хватало!)
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
        console.print(t("[bold red][ERROR] В папке ") + TARGET_FOLDER + t(" не найдено изображений![/bold red]"))
        return
        
    console.print(t("[bold cyan][ORAKUL SYSTEM][/bold cyan] [bold green]Найдено ") + str(len(image_paths)) + t(" холстов. Запуск глубокого анализа...[/bold green]"))
    
    with Progress(
        SpinnerColumn("dots", style="bold yellow"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(complete_style="green", finished_style="bold green", pulse_style="yellow"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[bold white]{task.completed}/{task.total}"),
        TextColumn(t("[dim]Время:[/dim]")),
        TimeElapsedColumn(),
        TextColumn(t("[dim]Осталось:[/dim]")),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task(t("Анализ холстов..."), total=len(image_paths))
        
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
                    
            except Exception as e:
                console.print(t("\n[bold red][ERROR] Ошибка на файле ") + str(img_path.name) + ": " + str(e) + "[/bold red]")

if __name__ == "__main__":
    main()