import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from rich.console import Console
from rich.panel import Panel

# Принудительно задаем UTF-8 для вывода в консоль Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

console = Console()

GITHUB_USER = 'OrakulStudio'
REPO_NAME = 'Dataset-Multitool'
REPO_ZIP_URL = (
    f'https://github.com/{GITHUB_USER}/{REPO_NAME}/archive/refs/heads/main.zip'
)


def t(text: str) -> str:
  """Функция перевода при отрисовке панели"""
  config_file = Path('config.json')
  lang = 'ru'

  if config_file.exists():
    try:
      with open(config_file, 'r', encoding='utf-8') as f:
        lang = json.load(f).get('language', 'ru')
    except Exception:
      pass

  if lang == 'ru' or not text:
    return text

  lang_file = Path(f'{lang}.json')
  if not lang_file.exists():
    return text

  try:
    with open(lang_file, 'r', encoding='utf-8') as f:
      raw_data = json.load(f)
      data = {k.replace('\ufe0f', ''): v for k, v in raw_data.items()}

    clean_text = text.replace('\ufe0f', '')

    if clean_text in data:
      return data[clean_text]

    if '\n' in clean_text:
      lines = clean_text.split('\n')
      translated_lines = [
          data.get(line.replace('\ufe0f', ''), line) for line in lines
      ]
      return '\n'.join(translated_lines)
  except Exception:
    pass

  return text


def get_local_commit_hash():
  """Получаем хэш текущего локального коммита юзера"""
  try:
    hash_val = (
        subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL
        )
        .decode('utf-8')
        .strip()
    )
    return hash_val
  except Exception:
    return None


def get_remote_commit_hash():
  """Получаем хэш последнего коммита с GitHub API (timeout 2 сек)"""
  url = f'https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/commits/main'
  try:
    req = urllib.request.Request(
        url, headers={'User-Agent': 'Dataset-Multitool-Updater'}
    )
    with urllib.request.urlopen(req, timeout=2) as response:
      if response.status == 200:
        data = json.loads(response.read().decode('utf-8'))
        return data['sha'][:7]
  except Exception:
    pass
  return None


def download_and_extract_full_repo(target_dir='.'):
  """Скачивает весь репозиторий пачкой через ZIP и обновляет файлы безопасно"""
  exclude_items = {
      'configs',
      'outputs',
      'datasets',
      '.git',
      'config.json',
  }  # Что нельзя затирать

  with tempfile.TemporaryDirectory() as temp_dir:
    zip_path = os.path.join(temp_dir, 'update.zip')

    # Красивый анимированный статус вместо замершего текста
    with console.status(
        '[bold yellow]⏳ |Uploading a new archive|Загрузка свежего архива репозитория с'
        ' GitHub...[/bold yellow]',
        spinner='dots',
    ):
      try:
        urllib.request.urlretrieve(REPO_ZIP_URL, zip_path)
      except Exception as e:
        console.print(f'[bold red]❌ Ошибка скачивания: {e}[/bold red]')
        return False

    with console.status(
        '[bold yellow]📦 Распаковка и подготовка файлов |Unpacking files|...[/bold yellow]',
        spinner='dots',
    ):
      try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
          zip_ref.extractall(temp_dir)
      except Exception as e:
        console.print(f'[bold red]❌ Ошибка распаковки: {e}[/bold red]')
        return False

    extracted_folders = [
        f
        for f in os.listdir(temp_dir)
        if os.path.isdir(os.path.join(temp_dir, f)) and f != '__MACOSX'
    ]
    if not extracted_folders:
      console.print('[bold red]❌ Ошибка структуры архива.[/bold red]')
      return False

    source_dir = os.path.join(
        temp_dir, extracted_folders[0], 'Orakul_Studio_Core'
    )
    if not os.path.exists(source_dir):
      source_dir = os.path.join(temp_dir, extracted_folders[0])

    with console.status(
        '[bold yellow]📂 Безопасное обновление |Safe update| файлов пачкой...[/bold yellow]',
        spinner='dots',
    ):
      for item in os.listdir(source_dir):
        if item in exclude_items:
          continue

        s_path = os.path.join(source_dir, item)
        d_path = os.path.join(target_dir, item)

        try:
          if os.path.isdir(s_path):
            if os.path.exists(d_path):
              shutil.rmtree(d_path)
            shutil.copytree(s_path, d_path)
          else:
            shutil.copy2(s_path, d_path)
        except Exception as e:
          console.print(f'[yellow]⚠️ Не удалось обновить|Failed to update {item}: {e}[/yellow]')

  return True


def print_banner():
  console.print(r""" [bold yellow]                                
 ░█▀█░█▀▄░█▀█░█░█░█░█░█░░░░░█▀▀░▀█▀░█░█░█▀▄░▀█▀░█▀█
 ░█░█░█▀▄░█▀█░█▀▄░█░█░█░░░░░▀▀█░░█░░█░█░█░█░█░░█░█
 ░▀▀▀░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░░░▀▀▀░░▀░░▀▀▀░▀▀░░▀▀▀░▀▀▀                                                                                                                                                                                                              
    """)


def draw_whats_new():
  changelog = (
      '[bold cyan]ℹ️ ЧТО НОВОГО В ВЕРСИИ / WHAT`S NEW:[/bold cyan]\n\n'
      '  [bold]— Qwen-VL Engine —[/bold]\n'
      '  [dim]RU:[/dim] Поддержка 3 моделей (2.5-VL-7B, 3-VL-8B, 3-VL-30B-FP8).\n'
      '  [dim]EN:[/dim] Support for 3 models (2.5-VL-7B, 3-VL-8B,'
      ' 3-VL-30B-FP8).\n'
      '  [yellow][TIP][/yellow] Qwen 2.5-VL: Рекомендуется / Recommended.\n\n'
      '  [bold]— Crop & Rename —[/bold]\n'
      '  [dim]RU:[/dim] Опциональный префикс (сохранение родных имён).\n'
      '  [dim]EN:[/dim] Optional prefix (keeping original names).\n'
      '  [dim]RU:[/dim] Умная синхронизация парных .txt файлов (игнор кэша).\n'
      '  [dim]EN:[/dim] Smart sync for paired .txt files (cache ignored).\n\n'
      '  [bold]— Z-Image SAEP-SVD —[/bold]\n'
      '  [dim]RU:[/dim] Интегрирован авторский алгоритм дистилляции (Orakul'
      ' Studio).\n'
      '  [dim]EN:[/dim] Integrated custom distillation algorithm (Orakul'
      ' Studio).\n'
      '  [yellow][TIP] SAEP Formula:[/yellow]\n'
      '  [dim]RU:[/dim] Sign Alignment (фаза) + Energy Preservation (инъекция'
      ' энергии).\n'
      '  [dim]EN:[/dim] Sign Alignment (phase) + Energy Preservation (energy'
      ' injection).\n'
      '  [green][INFO][/green]\n'
      '  [dim]RU:[/dim] Без потери микроконтраста (для r=2048 и стандартных'
      ' рангов).\n'
      '  [dim]EN:[/dim] Zero microcontrast loss (for r=2048 and standard'
      ' ranks).\n'
      '  [magenta][DOCS][/magenta] Технический отчет / Technical report:\n'
      '  [white]https://zenodo.org/records/23290969[/white]\n\n'
      '  [bold]— UI & Optimization —[/bold]\n'
      '  [dim]RU:[/dim] Меню оптимизировано: блок обновлений вынесен в'
      ' загрузчик.\n'
      '  [dim]EN:[/dim] UI optimized: update block moved to the startup'
      ' loader.\n'
      '  [dim]RU:[/dim] Обновлено меню: [L] вместо [LANG] для удобства'
      ' ввода.\n'
      '  [dim]EN:[/dim] UI updated: [L] instead of [LANG] for easier input.'
  )

  console.print(
      Panel(
          changelog,
          title='[bold yellow]🛠️ ORAKUL DATASET MULTITOOL[/bold yellow]',
          title_align='left',
          subtitle='[bold cyan]Update:12.10.26|v2.1.0[/bold cyan]',
          subtitle_align='right',
          border_style='cyan',
          expand=False,
      )
  )


def check_for_updates():
  local_hash = get_local_commit_hash()
  if not local_hash:
    return

  remote_hash = get_remote_commit_hash()
  if not remote_hash:
    return

  if local_hash != remote_hash:
    console.print('\n' + '=' * 55)
    print_banner()
    draw_whats_new()
    console.print(
        f'🚀[bold yellow] New features are available|Доступны новые фичи на GitHub! (Commit:'
        f' {remote_hash})[/bold yellow]'
    )
    console.print('=' * 55)

    answer = (
        input(
            'Обновить |Update| Dataset-Multitool прямо сейчас |right now|?'
            ' (y/n): '
        )
        .strip()
        .lower()
    )
    if answer == 'y':
      print('\n[+] Запускаем полную синхронизацию репозитория...')
      success = download_and_extract_full_repo()
      if success:
        print('[✓] Всё успешно обновлено пачкой! Перезапустите инструмент.')
        sys.exit(0)
      else:
        print('[!] Ошибка при обновлении файлов.')


if __name__ == '__main__':
  check_for_updates()