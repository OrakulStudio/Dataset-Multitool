# ==============================================================================
#  ORAKUL CORE — AI Dataset & Model MultiTool
#  Module: Model Poem
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

# ХАРДКОРНОЕ ПЕРЕНАПРАВЛЕНИЕ КЭША
# === ДИНАМИЧЕСКОЕ ПЕРЕНАПРАВЛЕНИЕ КЭША (UNIFIED PATHS) ===
config_file_cache = Path(__file__).parent / "paths.json"
cache_path = None

# 1. Если paths.json есть, читаем и ПРОВЕРЯЕМ диск
if config_file_cache.exists():
    try:
        with open(config_file_cache, "r", encoding="utf-8") as f:
            saved_path = json.load(f).get("hf_cache_dir", r"XXX:\AI_Models\Cache")
            drive = Path(saved_path).drive

            if drive and Path(f"{drive}/").exists():
                cache_path = saved_path
    except Exception:
        pass

# 2. Если конфига не было ИЛИ сохранённого диска нет на этом ПК
if not cache_path:
    cache_path = r"XXX:\AI_Models\Cache"
    if not Path("XXX:/").exists():
        print("\n[!] Диск по умолчанию не обнаружен.")
        user_drive = (
            input("Укажите свою букву диска для скачивания Qwen (например, C, D, E): ")
            .strip()
            .upper()
        )
        if user_drive:
            user_drive = user_drive[0]
            cache_path = rf"{user_drive}:\AI_Models\Cache"

    # Перезаписываем paths.json
    current_config = {}
    if config_file_cache.exists():
        try:
            with open(config_file_cache, "r", encoding="utf-8") as fr:
                current_config = json.load(fr)
        except Exception:
            pass
            
    current_config["hf_cache_dir"] = cache_path
    with open(config_file_cache, "w", encoding="utf-8") as f:
        json.dump(current_config, f, indent=4)

# Фиксируем кэш для HuggingFace до загрузки библиотек
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HOME"] = cache_path
# =========================================================

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Инициализация красивой консоли
console = Console()

# =====================================================================
# НАСТРОЙКА МОДЕЛИ И ПРИЁМ ДАННЫХ ОТ ПУЛЬТА
# =====================================================================
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# Проверяем, что скрипт запущен через наш пульт run.py
if len(sys.argv) < 3:
    console.print(t("[bold red][ERROR] Скрипт нужно запускать через пульт run.py![/bold red]"))
    sys.exit(1)

TARGET_FOLDER = sys.argv[1] 
PROMPT_FILE = sys.argv[2]

# Читаем текст промпта из файла, сохраненного в Notepad++
try:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read().strip()
except Exception as e:
    console.print(t("[bold red][ERROR] Ошибка чтения промпта: ") + f"{e}[/bold red]")
    sys.exit(1)
# =====================================================================

def main():
    console.print(t("[bold cyan][ORAKUL SYSTEM][/bold cyan] [bold yellow]Запуск генератора лора на базе: ") + MODEL_NAME + t("...[/bold yellow]"))
    
    # Оборачиваем загрузку весов в красивый статус
    with console.status(t("[bold yellow]Загрузка весов в видеопамять CUDA...[/bold yellow]"), spinner="dots"):
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda"
        )
    
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.PNG', '.JPG', '.JPEG')
    image_paths = sorted([p for p in Path(TARGET_FOLDER).glob("*") if p.suffix in valid_exts])
    
    if not image_paths:
        console.print(t("[bold red][ERROR] В папке ") + TARGET_FOLDER + t(" не найдено изображений![/bold red]"))
        return
        
    # Сборка холстов со спиннером
    with Progress(
        SpinnerColumn("dots", style="bold yellow"),
        TextColumn(t("[bold cyan]Загрузка холстов в единый контекст...[/bold cyan]")),
        console=console
    ) as progress:
        progress.add_task("loading", total=None)
        
        content_list = []
        for img_path in image_paths:
            content_list.append({"type": "image", "image": str(img_path.resolve())})
        
        content_list.append({"type": "text", "text": SYSTEM_PROMPT})
        
        messages = [
            {
                "role": "user",
                "content": content_list
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
    
    # ВОТ ЗДЕСЬ БЫЛ ТАЙМАУТ БЕЗ ИНДИКАТОРА: добавляем анимацию работы нейросети
    with console.status(t("[bold magenta]Нейросеть пишет лор-документ (генерация токенов)...[/bold magenta]"), spinner="dots"):
        with torch.no_grad():
            # До 2048 токенов на целостный сюжет
            generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    caption = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Сохраняем в один итоговый файл
    output_file = Path(TARGET_FOLDER) / "ORAKUL_LORE.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(caption.strip())
        
    console.print(t("\n[bold green][SUCCESS][/bold green] Лор-документ успешно создан и сохранен в: [bold white]") + str(output_file) + t("[/bold white]\n"))
    
    # Выводим фрагмент текста в красивой рамке
    preview_text = caption[:1000] + "..." if len(caption) > 1000 else caption
    console.print(Panel(preview_text, title=t("[bold cyan]ФРАГМЕНТ СГЕНЕРИРОВАННОГО ЛОРА[/bold cyan]"), border_style="cyan"))

if __name__ == "__main__":
    main()