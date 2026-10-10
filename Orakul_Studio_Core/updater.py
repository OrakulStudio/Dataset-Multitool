import subprocess
import urllib.request
import json
import sys
from rich.console import Console
from rich.panel import Panel

console = Console()

GITHUB_USER = "OrakulStudio"
REPO_NAME = "Dataset-Multitool"

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
        console.print(f"🚀[bold yellow] Доступны новые фичи на GitHub! (Commit: {remote_hash})")
        console.print("="*55)
        
        answer = input("Обновить Dataset-Multitool прямо сейчас? (y/n): ").strip().lower()
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