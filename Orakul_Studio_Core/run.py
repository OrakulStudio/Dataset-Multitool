# ==============================================================================
#  ORAKUL CORE — AI Dataset & Model MultiTool
#  Module: Main Studio Hub & Interactive CLI Control Center
# ------------------------------------------------------------------------------
#  Author:      Orakul (Orakul Studio)
#  GitHub:      https://github.com/OrakulStudio
#  Civitai:     https://civitai.com/user/ORAKUL_STUDIO
#  HuggingFace: https://huggingface.co/OrakulStorm
# ------------------------------------------------------------------------------
#  Description: All-in-One local AI automation pipeline launcher.
#  Copyright (c) 2026 Orakul Studio. All rights reserved.
# ==============================================================================
import os
import sys
import json
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

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
            # 1. Зачищаем невидимый символ во всех ключах словаря
            data = {k.replace('\ufe0f', ''): v for k, v in raw_data.items()}
            
        # 2. Зачищаем невидимый символ во входящем тексте
        clean_text = text.replace('\ufe0f', '')
            
        # Прямое точное совпадение
        if clean_text in data:
            return data[clean_text]
            
        # Если это многострочный текст (меню), переводим каждую строку отдельно
        if "\n" in clean_text:
            lines = clean_text.split("\n")
            translated_lines = [data.get(line.replace('\ufe0f', ''), line) for line in lines]
            return "\n".join(translated_lines)
    except Exception:
        pass
        
    return text

# Инициализация красивой консоли
console = Console()

# --- АБСОЛЮТНО ПОРТАТИВНЫЕ ПУТИ (Относительно папки Orakul_Studio_Core) ---
BASE_DIR = Path(__file__).resolve().parent

NPP_PATH = BASE_DIR / "Notepad" / "notepad++.exe"
VENV_PYTHON = BASE_DIR / "orakul_env" / "Scripts" / "python.exe"

VIKING_SCRIPT = BASE_DIR / "viking_caption_qwen.py"
POEM_SCRIPT = BASE_DIR / "poem.py"
SCAN_SCRIPT = BASE_DIR / "clode_universal_scan.py"
CLEAN_SCRIPT = BASE_DIR / "auto_clean_total.py"
CROP_SCRIPT = BASE_DIR / "crop_and_rename.py"
META_SCRIPT = BASE_DIR / "fix_metadata.py"
MERGE_SCRIPT = BASE_DIR / "merge_manager.py"
DATASET_SCRIPT = BASE_DIR / "dataset_manager.py"
CHECK_SCRIPT = BASE_DIR / "check_params.py"

# --- ПРОВЕРКА ВИРТУАЛЬНОГО ОКРУЖЕНИЯ  ---
def check_venv():
    if not VENV_PYTHON.exists():
        console.print(t("\n[bold red][ERROR] Виртуальное окружение 'orakul_env' не найдено | Virtual environment not found![/bold red]"))
        console.print(t("\n[yellow]Выполни по порядку в этой же папке | Perform the steps in order within the same folder. :[/yellow]\n"))
        console.print("  pip install -r requirements.txt")
        console.print("  python -m venv orakul_env")
        console.print("  orakul_env\\Scripts\\activate  (в Git Bash / Linux: source orakul_env/Scripts/activate)")
        console.print("  pip install -r requirements_caption_env.txt")
        console.print("  deactivate")
        console.print("  python run.py")
        console.print(t("\n[dim]Нажми Enter для выхода(Press Enter to exit)...[/dim]"))
        input()
        sys.exit(1)
# ---------------------------------------------------------


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_qwen_task(script_path, task_name):
    console.print(f"\n[bold yellow]--- {task_name.upper()} ---[/bold yellow]")
    
    target_folder = console.input(t("[bold cyan]Перетащи папку с картинками сюда:[/bold cyan] ")).strip(' "\'')
    
    if not os.path.exists(target_folder):
        console.print(t("[bold red][-] Ошибка: Папка не найдена![/bold red]"))
        return

    temp_prompt_file = BASE_DIR / "CURRENT_ACTIVE_PROMPT.txt"
    
    console.print("\n" + t("[bold magenta][!][/bold magenta] [white]Открываю Notepad++. Вставь промпт,[/white] [bold green]СОХРАНИ (Ctrl+S)[/bold green] [white]и[/white] [bold red]ЗАКРОЙ окно![/bold red]"))
    
    subprocess.run([str(NPP_PATH), "-multiInst", "-nosession", str(temp_prompt_file)])
    
    if not os.path.exists(temp_prompt_file) or os.path.getsize(temp_prompt_file) == 0:
        console.print(t("[bold red][-] Промпт пустой! Отмена задачи.[/bold red]"))
        return

    console.print("\n" + t("[bold green][+][/bold green] [bold white]Запускаем нейронку. Память выделена...[/bold white]"))
    subprocess.run([str(VENV_PYTHON), str(script_path), target_folder, str(temp_prompt_file)])
    console.print("\n" + t("[bold green][+][/bold green] [bold cyan]Процесс завершен! Видеопамять полностью освобождена.[/bold cyan]"))

def run_simple_task(script_path, task_name):
    console.print(f"\n[bold yellow]--- {task_name.upper()} ---[/bold yellow]")
    target_folder = console.input(t("[bold cyan]Перетащи папку с датасетом сюда:[/bold cyan] ")).strip(' "\'')
    
    if not os.path.exists(target_folder):
        console.print(t("[bold red][-] Ошибка: Папка не найдена![/bold red]"))
        return

    console.print(f"\n[bold green][+][/bold green] [bold white]{t('Запуск утилиты')} {task_name}...[/bold white]")
    subprocess.run([str(VENV_PYTHON), str(script_path), target_folder])
    console.print(f"\n[bold green][+][/bold green] [bold cyan]{t('Задача')} '{task_name}' {t('успешно выполнена!')}[/bold cyan]")

def run_interactive_task(script_path, task_name):
    console.print(f"\n[bold yellow]--- {task_name.upper()} ---[/bold yellow]")
    console.print(f"\n[bold green][+][/bold green] [bold white]{t('Запуск утилиты')} {task_name}...[/bold white]")
    subprocess.run([str(VENV_PYTHON), str(script_path)])
    console.print(f"\n[bold green][+][/bold green] [bold cyan]{t('Задача')} '{task_name}' {t('успешно выполнена!')}[/bold cyan]")

def set_current_lang(lang_code):
    config_file = Path("config.json")
    config_data = {}
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception:
            pass
    config_data["language"] = lang_code
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)

def main():
    check_venv()  # <-- один раз при запуске
    while True:
        clear_screen()
        
        menu_text = (
            "[bold cyan][1][/bold cyan] Viking Caption (Техническая разметка датасета)\n"
            "[bold cyan][2][/bold cyan] Qwen Poem (Создание лора по изображениям)\n"
            "[bold cyan][3][/bold cyan] Caption QC Scanner (Сканирование ошибок и мусора)\n"
            "[bold cyan][4][/bold cyan] Auto Clean Dataset (Автоочистка по CSV-отчёту)\n"
            "[bold cyan][5][/bold cyan] Crop and Rename (Ресайз и переименование)\n"
            "[bold cyan][6][/bold cyan] Metadata Injector (Перенос воркфлоу/промптов в ретушь)\n"
            "[bold cyan][7][/bold cyan] Weight Merger (Слияние весов Flux.2 & Z-Image)\n"
            "[bold cyan][8][/bold cyan] Dataset Manager (Аугментация датасета & Нумерация)\n"
            "[bold cyan][9][/bold cyan] Model Inspector (Параметры & Триггерные слова)\n"
            "[bold red][0][/bold red] Выход\n\n"
            "[bold magenta][LANG][/bold magenta] Смена языка (Language)"
        )
        
        console.print(Panel(
            t(menu_text), 
            title=f"[bold yellow]{t('⚡ ORAKUL STUDIO — WORKFLOW MENU ⚡')}[/bold yellow]", 
            border_style="yellow",
            expand=False
        ))

        choice = console.input("\n" + t("[bold white]Выбери действие (0-9, lang): [/bold white]")).strip().lower()

        if choice == '1':
            run_qwen_task(VIKING_SCRIPT, "Viking Caption")
            console.input("\n" + t("[dim]Нажми Enter для возврата в меню...[/dim]"))
        elif choice == '2':
            run_qwen_task(POEM_SCRIPT, "Qwen Poem")
            console.input("\n" + t("[dim]Нажми Enter для возврата...[/dim]"))
        elif choice == '3':
            run_simple_task(SCAN_SCRIPT, "QC Scanner")
            console.input("\n" + t("[dim]Нажми Enter для возврата...[/dim]"))
        elif choice == '4':
            run_simple_task(CLEAN_SCRIPT, "Dataset Cleaner")
            console.input("\n" + t("[dim]Нажми Enter для возврата...[/dim]"))
        elif choice == '5':
            run_simple_task(CROP_SCRIPT, "Crop and Rename")
            console.input("\n" + t("[dim]Нажми Enter для возврата...[/dim]"))
        elif choice == '6':
            run_interactive_task(META_SCRIPT, "Metadata Injector")
            console.input("\n" + t("[dim]Нажми Enter для возврата...[/dim]"))
        elif choice == '7':
            run_interactive_task(MERGE_SCRIPT, "Weight Merger")
            console.input("\n" + t("[dim]Нажми Enter для возврата...[/dim]"))
        elif choice == '8':
            run_interactive_task(DATASET_SCRIPT, "Dataset Manager")
            console.input("\n" + t("[dim]Нажми Enter для возврата...[/dim]"))
        elif choice == '9':
            run_interactive_task(CHECK_SCRIPT, "Model Inspector")
            console.input("\n" + t("[dim]Нажми Enter для возврата...[/dim]")) 
        elif choice in ['lang', 'l']:
            new_lang = console.input("\n" + t("[bold cyan]Выбери язык / Choose language (ru, en): [/bold cyan]")).strip().lower()
            if new_lang in ['ru', 'en']:
                set_current_lang(new_lang)
                console.print(t("[bold green][+] Язык изменен на:[/bold green]") + f" [bold green]{new_lang.upper()}[/bold green]")
            else:
                console.print(t("[bold red][-] Неверный язык![/bold red]"))
                console.input("\n" + t("[dim]Нажми Enter для возврата...[/dim]"))
        elif choice == '0':
            console.print(t("Выход из системы. Удачного обучения!"))
            break

if __name__ == "__main__":
    main()