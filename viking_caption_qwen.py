# ==============================================================================
#  ORAKUL CORE — AI Dataset & Model MultiTool
#  Module: Viking Caption (Qwen2.5-VL Auto-Captioner)
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

# === ДИНАМИЧЕСКОЕ ПЕРЕНАПРАВЛЕНИЕ КЭША (PORTABLE-READY) ===
config_file_cache = Path(__file__).parent / "paths.json"
cache_path = None

# 1. Если config.json есть, читаем и ПРОВЕРЯЕМ, существует ли этот диск на текущем ПК
if config_file_cache.exists():
    try:
        with open(config_file_cache, "r", encoding="utf-8") as f:
            saved_path = json.load(f).get("hf_cache_dir", r"xxx:\AI_Models\Cache")
            drive = Path(saved_path).drive  # Достаёт букву диска, например "E:"

            # Если диск из сохранённого конфига есть в системе — берем его
            if drive and Path(f"{drive}/").exists():
                cache_path = saved_path
            else:
                print(
                    t("\n[!] Диск ") + drive + t(" из paths.json не найден на этой машине.")
                )
    except Exception:
        pass

# 2. Если конфига не было ИЛИ сохранённого диска нет на этом ПК
if not cache_path:
    cache_path = r"xxx:\AI_Models\Cache"
    if not Path("xxx:/").exists():
        print(t("\n[!] Указанный путь не обнаружен (Диск  отсутствует)."))
        user_drive = (
            input(
                t("Укажите свою букву диска для скачивания Qwen (например, C, E, G): ")
            )
            .strip()
            .upper()
        )
        if user_drive:
            cache_path = rf"{user_drive}:\AI_Models\Cache"

    # Перезаписываем ("w") config.json актуальным диском текущего ПК
    with open(config_file_cache, "w", encoding="utf-8") as f:
        # Сначала читаем существующий конфиг, чтобы не затереть язык
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

import torch
from PIL import Image
# =====================================================================
# ЖЕСТКИЙ ИМПОРТ: Вызываем архитектуру Qwen напрямую, без Auto-рулетки
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
# =====================================================================
from qwen_vl_utils import process_vision_info
from pathlib import Path
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

# =====================================================================
# НАСТРОЙКА МОДЕЛИ И ПРИЁМ ДАННЫХ ОТ ПУЛЬТА
# =====================================================================
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# Проверяем, что скрипт запущен через наш пульт
if len(sys.argv) < 3:
    console.print(t("[bold red][ERROR] Скрипт нужно запускать через пульт run.py![/bold red]"))
    sys.exit(1)

TARGET_FOLDER = sys.argv[1] 
PROMPT_FILE = sys.argv[2]

try:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read().strip()
except Exception as e:
    console.print(t("[bold red][ERROR] Ошибка чтения промпта: ") + f"{e}[/bold red]")
    sys.exit(1)
# =====================================================================

def main():
    console.print(t("[bold cyan][ORAKUL SYSTEM][/bold cyan] [bold yellow]Инициализация интеллектуального монстра: ") + MODEL_NAME + t("...[/bold yellow]"))
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    # Загрузка модели напрямую через её родной класс + обход flash_attn
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda"
    )
    
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.PNG', '.JPG', '.JPEG')
    image_paths = [p for p in Path(TARGET_FOLDER).glob("*") if p.suffix in valid_exts]
    
    if not image_paths:
        console.print(t("[bold red][ERROR] В папке ") + TARGET_FOLDER + t(" не найдено изображений![/bold red]"))
        return
        
    console.print(t("[bold cyan][ORAKUL SYSTEM][/bold cyan] [bold green]Найдено ") + str(len(image_paths)) + t(" холстов. Запуск глубокого анализа...[/bold green]"))
    
    # === НАСТРОЙКА КРАСИВОГО ПРОГРЕСС-БАРА ===
    with Progress(
        SpinnerColumn("dots", style="bold yellow"),  # Те самые крутящиеся песчинки
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(complete_style="green", finished_style="bold green", pulse_style="yellow"), # Тонкая полоса
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
                    
                # Двигаем наш новый прогресс-бар на 1 шаг вперед
                progress.advance(task)
                    
            except Exception as e:
                console.print(t("\n[bold red][ERROR] Ошибка на файле ") + str(img_path.name) + ": " + str(e) + "[/bold red]")

if __name__ == "__main__":
    main()