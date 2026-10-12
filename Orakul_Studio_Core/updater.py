import subprocess
import urllib.request
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

GITHUB_USER = "OrakulStudio"
REPO_NAME = "Dataset-Multitool"

def t(text: str) -> str:
    """Функция перевода  при отрисовке панели"""
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

def get_local_commit_hash():
    """Получаем хэш текущего локального коммита юзера"""
    try:
        hash_val = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], 
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        return hash_val
    except Exception:
        return None

def get_remote_commit_hash():
    """Получаем хэш последнего коммита с GitHub API (timeout 2 сек)"""
    url = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/commits/main"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Dataset-Multitool-Updater'})
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                # Возвращаем первые 7 символов хэша
                return data['sha'][:7]
    except Exception:
        pass
    return None
    
def print_banner():
    console.print(r""" [bold yellow]                                
 ░█▀█░█▀▄░█▀█░█░█░█░█░█░░░░░█▀▀░▀█▀░█░█░█▀▄░▀█▀░█▀█
 ░█░█░█▀▄░█▀█░█▀▄░█░█░█░░░░░▀▀█░░█░░█░█░█░█░░█░░█░█
 ░▀▀▀░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░░░▀▀▀░░▀░░▀▀▀░▀▀░░▀▀▀░▀▀▀                                                                                                                                                                                                              
    """)     

def draw_whats_new():
    changelog = (
        "[bold cyan]ℹ️ ЧТО НОВОГО В ВЕРСИИ / WHAT'S NEW:[/bold cyan]\n\n"
        
        "  [bold]— Qwen-VL Engine —[/bold]\n"
        "  [dim]RU:[/dim] Поддержка 3 моделей (2.5-VL-7B, 3-VL-8B, 3-VL-30B-FP8).\n"
        "  [dim]EN:[/dim] Support for 3 models (2.5-VL-7B, 3-VL-8B, 3-VL-30B-FP8).\n"
        "  [yellow][TIP][/yellow] Qwen 2.5-VL: Рекомендуется / Recommended.\n\n"
        
        "  [bold]— Crop & Rename —[/bold]\n"
        "  [dim]RU:[/dim] Опциональный префикс (сохранение родных имён).\n"
        "  [dim]EN:[/dim] Optional prefix (keeping original names).\n"
        "  [dim]RU:[/dim] Умная синхронизация парных .txt файлов (игнор кэша).\n"
        "  [dim]EN:[/dim] Smart sync for paired .txt files (cache ignored).\n\n"
        
        "  [bold]— Z-Image SAEP-SVD —[/bold]\n"
        "  [dim]RU:[/dim] Интегрирован авторский алгоритм дистилляции (Orakul Studio).\n"
        "  [dim]EN:[/dim] Integrated custom distillation algorithm (Orakul Studio).\n"
        "  [yellow][TIP] SAEP Formula:[/yellow]\n"
        "  [dim]RU:[/dim] Sign Alignment (фаза) + Energy Preservation (инъекция энергии).\n"
        "  [dim]EN:[/dim] Sign Alignment (phase) + Energy Preservation (energy injection).\n"
        "  [green][INFO][/green]\n"
        "  [dim]RU:[/dim] Без потери микроконтраста (для r=2048 и стандартных рангов).\n"
        "  [dim]EN:[/dim] Zero microcontrast loss (for r=2048 and standard ranks).\n"
        "  [magenta][DOCS][/magenta] Технический отчет / Technical report:\n"
        "  [white]https://zenodo.org/records/23290969[/white]\n\n"
        
        "  [bold]— UI & Optimization —[/bold]\n"
        "  [dim]RU:[/dim] Меню оптимизировано: блок обновлений вынесен в загрузчик.\n"
        "  [dim]EN:[/dim] UI optimized: update block moved to the startup loader.\n"
        "  [dim]RU:[/dim] Обновлено меню: [L] вместо [LANG] для удобства ввода.\n"
        "  [dim]EN:[/dim] UI updated: [L] instead of [LANG] for easier input."
    )
    
    console.print(Panel(
        changelog,
        title="[bold yellow]🛠️ ORAKUL DATASET MULTITOOL[/bold yellow]",
        title_align="left",
        subtitle="[bold cyan]Update:12.10.26|v2.1.0[/bold cyan]", 
        subtitle_align="right",
        border_style="cyan",
        expand=False
    ))

def check_for_updates():
    local_hash = get_local_commit_hash()
    if not local_hash:
        return  # У юзера нет git-папки или произошла ошибка

    remote_hash = get_remote_commit_hash()
    if not remote_hash:
        return  # Нет инета или GitHub не ответил
    
    # Если хэши НЕ совпадают — значит на GitHub лежит свежий код!
    if local_hash != remote_hash:
        console.print("\n" + "="*55)
        print_banner()
        draw_whats_new()
        console.print(f"🚀[bold yellow] Доступны новые фичи на GitHub! (Commit: {remote_hash})[/bold yellow]")
        console.print("="*55)
        
        answer = input("Обновить |Update| Dataset-Multitool прямо сейчас |right now|? (y/n): ").strip().lower()
        if answer == 'y':
            print("\n[+] Подтягиваем изменения с GitHub...")
            try:
                subprocess.run(["git", "pull"], check=True)
                print("[✓] Код успешно обновлен! Перезапустите инструмент.")
                sys.exit(0)
            except Exception as e:
                print(f"[!] Ошибка обновления: {e}")
                      

if __name__ == "__main__":
    check_for_updates()