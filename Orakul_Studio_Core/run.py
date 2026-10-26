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
import time
from pathlib import Path

if os.name == 'nt':
    import msvcrt

from rich.console import Console, Group
from rich.live import Live
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


def check_venv():
    if not VENV_PYTHON.exists():
        console.print(t("\n[bold white]\\[ERROR][/bold white] [dim]Virtual environment 'orakul_env' not found![/dim]"))
        console.print(t("\n[grey74]Run the following commands in this folder:[/grey74]\n"))
        console.print("  pip install -r requirements.txt")
        console.print("  python -m venv orakul_env")
        console.print("  orakul_env\\Scripts\\activate")
        console.print("  pip install -r requirements_caption_env.txt")
        console.print("  deactivate")
        console.print("  python run.py")
        input(t("\n[dim]Press Enter to exit...[/dim]"))
        sys.exit(1)


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    console.print("\n  [bold white]ORAKUL STUDIO[/bold white] [dim]— AI DATASET & MODEL MULTITOOL[/dim]\n")


def run_qwen_task(script_path, task_name):
    print_banner()
    console.print(f"[bold white]\\[RUNNING][/bold white] [grey74]{task_name.upper()}[/grey74]")
    console.print(t("\n[bold white]Select Qwen-VL version:[/bold white]"))
    console.print(t("  [bold white][ 1 ][/bold white] [white]Qwen 2.5-VL-7B-Instruct[/white]       [dim](~15 GB VRAM)[/dim]"))
    console.print(t("  [bold white][ 2 ][/bold white] [white]Qwen3-VL-8B-Instruct[/white]          [dim](~18 GB VRAM)[/dim]"))
    console.print(t("  [bold white][ 3 ][/bold white] [white]Qwen3-VL-30B-A3B-Instruct-FP8[/white] [dim](~32 GB VRAM / Heavy artillery)[/dim]"))

    model_choice = console.input(t("\n[bold white]> Your choice (1-3) [Enter=1]: [/bold white]")).strip()

    if model_choice == '3':
        model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
        console.print(t("[bold white]\\[MODEL][/bold white] [dim]Selected powerful Qwen3-VL-30B-FP8.[/dim]"))
    elif model_choice == '2':
        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        console.print(t("[bold white]\\[MODEL][/bold white] [dim]Selected Qwen3-VL-8B-Instruct.[/dim]"))
    else:
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        console.print(t("[bold white]\\[MODEL][/bold white] [dim]Selected proven Qwen 2.5-VL-7B.[/dim]"))

    target_folder = console.input(t("\n[bold white]> Drag and drop the folder with images here:[/bold white] ")).strip(' "\'')
    if not os.path.exists(target_folder):
        console.print(t("[bold white]\\[ERROR][/bold white] [dim]Folder not found![/dim]"))
        return

    temp_prompt_file = BASE_DIR / "CURRENT_ACTIVE_PROMPT.txt"
    console.print("\n" + t("[bold white]\\[ACTION][/bold white] [grey74]Opening Notepad++. Paste the prompt, SAVE (Ctrl+S) and CLOSE the window.[/grey74]"))
    subprocess.run([str(NPP_PATH), "-multiInst", "-nosession", str(temp_prompt_file)])

    if not os.path.exists(temp_prompt_file) or os.path.getsize(temp_prompt_file) == 0:
        console.print(t("[bold white]\\[ERROR][/bold white] [dim]Prompt is empty! Task cancelled.[/dim]"))
        return

    console.print("\n" + t("[bold white]\\[SYSTEM][/bold white] [dim]Starting neural network. Memory allocated...[/dim]"))
    subprocess.run([str(VENV_PYTHON), str(script_path), target_folder, str(temp_prompt_file), model_id])
    console.print("\n" + t("[bold white]\\[DONE][/bold white] [dim]Process finished. VRAM released.[/dim]"))


def run_simple_task(script_path, task_name):
    print_banner()
    console.print(f"[bold white]\\[RUNNING][/bold white] [grey74]{task_name.upper()}[/grey74]")
    target_folder = console.input(t("\n[bold white]> Drag and drop the dataset folder here:[/bold white] ")).strip(' "\'')
    if not os.path.exists(target_folder):
        console.print(t("[bold white]\\[ERROR][/bold white] [dim]Folder not found![/dim]"))
        return
    console.print(f"\n[bold white]\\[SYSTEM][/bold white] [dim]{t('Starting utility')} {task_name}...[/dim]")
    subprocess.run([str(VENV_PYTHON), str(script_path), target_folder])
    console.print(f"\n[bold white]\\[DONE][/bold white] [dim]{t('Task')} '{task_name}' {t('completed successfully!')}[/dim]")


def run_interactive_task(script_path, task_name):
    print_banner()
    console.print(f"[bold white]\\[RUNNING][/bold white] [grey74]{task_name.upper()}[/grey74]")
    console.print(f"\n[bold white]\\[SYSTEM][/bold white] [dim]{t('Starting utility')} {task_name}...[/dim]")
    subprocess.run([str(VENV_PYTHON), str(script_path)])
    console.print(f"\n[bold white]\\[DONE][/bold white] [dim]{t('Task')} '{task_name}' {t('completed successfully!')}[/dim]")


def clone_ai_toolkit():
    print_banner()
    console.print(f"[bold white]\\[RUNNING][/bold white] [grey74]{t('CLONE AI-TOOLKIT (WINDOWS 11)')}[/grey74]")
    repo_url = "https://github.com/OrakulStudio/AI-Toolkit-Windows11"
    user_input = console.input(t("\n[bold white]> Specify save path or drag a folder [Enter = current utility folder]: [/bold white]")).strip(' "\'')

    target_dir = BASE_DIR / "AI-Toolkit-Windows11" if not user_input else Path(user_input) / "AI-Toolkit-Windows11"

    if target_dir.exists() and os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        console.print(t(f"[bold white]\\[ERROR][/bold white] [dim]Folder already exists and is not empty:[/dim] [white]{target_dir}[/white]"))
        return

    console.print(f"\n[bold white]\\[SYSTEM][/bold white] [dim]{t('Cloning repository to:')}[/dim] [white]{target_dir}[/white]")
    try:
        subprocess.run(["git", "clone", repo_url, str(target_dir)], check=True)
        console.print(f"\n[bold white]\\[DONE][/bold white] [dim]{t('Repository cloned successfully!')}[/dim]")
    except Exception as e:
        console.print(f"\n[bold white]\\[ERROR][/bold white] [dim]{t('Error while cloning:')} {e}[/dim]")


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


def get_keypress():
    if os.name == 'nt' and msvcrt.kbhit():
        ch = msvcrt.getch()
        if ch in (b'\x00', b'\xe0'):
            msvcrt.getch()
            return None
        if ch == b'\x08':
            return 'backspace'
        if ch in (b'\r', b'\n'):
            return 'enter'
        try:
            return ch.decode('utf-8').lower()
        except Exception:
            return None
    return None


def generate_ui(input_buffer: str, status_msg: str) -> Group:
    header = Panel(
        "[bold white]ORAKUL STUDIO[/bold white] [dim]— AI DATASET & MODEL MULTITOOL[/dim]",
        border_style="grey50",
        box=box.ROUNDED
    )

    table = Table(box=box.SIMPLE_HEAD, expand=True, border_style="dim")
    table.add_column("CMD", justify="right", style="bold white", width=5)
    table.add_column("MODULE", style="white", ratio=1)
    table.add_column("DESCRIPTION", style="dim", ratio=2)

    table.add_row("1", "Viking Caption", t("Technical dataset captioning"))
    table.add_row("2", "Qwen Poem", t("Create lore from images"))
    table.add_row("3", "Caption QC Scanner", t("Scan for errors and junk"))
    table.add_row("4", "Auto Clean Dataset", t("Auto-clean by CSV report"))
    table.add_row("5", "Crop and Rename", t("Resize and rename"))
    table.add_row("6", "Metadata Injector", t("Transfer workflow/prompts to retouch"))
    table.add_row("7", "Weight Merger & SVD Lab", t("Weight merging & SVD distillation"))
    table.add_row("8", "Dataset Manager", t("Dataset augmentation & numbering"))
    table.add_row("9", "Model Inspector", t("Parameters & trigger words"))
    table.add_row("10", "AI-Toolkit Win11", t("Clone AI-Toolkit repository"))

    menu = Panel(
        table,
        title=t("[bold white] SYSTEM HUB v2.2.1 [/bold white]"),
        border_style="grey50",
        box=box.ROUNDED
    )

    footer_text = t("[dim]Actions:[/dim] [bold white][1-10][/bold white] [dim]Run Module |[/dim] [bold white][L][/bold white] [dim]Lang |[/dim] [bold white][0][/bold white] [dim]Quit[/dim]")
    prompt = f"[bold white]> Select action (0-10, l):[/bold white] [bold cyan]{input_buffer}█[/bold cyan]"
    if status_msg:
        prompt += f"   {status_msg}"

    footer = Panel(f"{footer_text}\n\n{prompt}", border_style="grey50", box=box.ROUNDED)

    return Group(header, menu, footer)


def main():
    check_venv()
    input_buffer = ""
    status_msg = ""

    with Live(generate_ui(input_buffer, status_msg), refresh_per_second=20, screen=True) as live:
        while True:
            key = get_keypress()

            if key:
                if key == 'backspace':
                    input_buffer = input_buffer[:-1]
                    status_msg = ""
                elif key == 'enter':
                    choice = input_buffer.strip()
                    input_buffer = ""

                    if choice == '0':
                        break

                    elif choice in [str(i) for i in range(1, 11)] or choice in ['l', 'lang']:
                        live.stop()
                        clear_screen()

                        if choice == '1':
                            run_qwen_task(VIKING_SCRIPT, "Viking Caption")
                        elif choice == '2':
                            run_qwen_task(POEM_SCRIPT, "Qwen Poem")
                        elif choice == '3':
                            run_simple_task(SCAN_SCRIPT, "QC Scanner")
                        elif choice == '4':
                            run_simple_task(CLEAN_SCRIPT, "Dataset Cleaner")
                        elif choice == '5':
                            run_simple_task(CROP_SCRIPT, "Crop and Rename")
                        elif choice == '6':
                            run_interactive_task(META_SCRIPT, "Metadata Injector")
                        elif choice == '7':
                            run_interactive_task(MERGE_SCRIPT, "Weight Merger")
                        elif choice == '8':
                            run_interactive_task(DATASET_SCRIPT, "Dataset Manager")
                        elif choice == '9':
                            run_interactive_task(CHECK_SCRIPT, "Model Inspector")
                        elif choice == '10':
                            clone_ai_toolkit()
                        elif choice in ['lang', 'l']:
                            print_banner()
                            new_lang = console.input("\n" + t("[bold white]> Choose language (ru, en): [/bold white]")).strip().lower()
                            if new_lang in ['ru', 'en']:
                                set_current_lang(new_lang)
                                console.print(t("[bold white]\\[DONE][/bold white] [dim]Language changed to:[/dim]") + f" [white]{new_lang.upper()}[/white]")
                            else:
                                console.print(t("[bold white]\\[ERROR][/bold white] [dim]Invalid language![/dim]"))

                        console.input("\n" + t("[dim]> Press Enter to return to menu...[/dim]"))
                        live.start()
                    else:
                        status_msg = t("[bold red]Unknown command[/bold red]")
                else:
                    input_buffer += key
                    status_msg = ""

            
            time.sleep(0.05)

    clear_screen()
    console.print(t("\n[dim]Exiting. Happy training![/dim]"))


if __name__ == "__main__":
 
    main()