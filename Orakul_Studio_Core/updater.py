# ==============================================================================
#  ORAKUL CORE — AI Dataset & Model MultiTool
#  Module: Auto Updater (Direct Zip-Sync)
# ------------------------------------------------------------------------------
#  Author:      Orakul (Orakul Studio)
#  GitHub:      https://github.com/OrakulStudio
#  Civitai:     https://civitai.com/user/ORAKUL_STUDIO
#  HuggingFace: https://huggingface.co/OrakulStorm
# ------------------------------------------------------------------------------
#  Description: Direct full-repository updater pulling latest archive from GitHub.
#  Copyright (c) 2026 Orakul Studio. All rights reserved.
# ==============================================================================

import io
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from rich.console import Console
from rich.panel import Panel
from rich import box

# Force UTF-8 encoding for Windows console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

console = Console()

GITHUB_USER = 'OrakulStudio'
REPO_NAME = 'Dataset-Multitool'
REPO_ZIP_URL = f'https://github.com/{GITHUB_USER}/{REPO_NAME}/archive/refs/heads/main.zip'


def download_and_extract_full_repo(target_dir='.'):
    """Downloads entire repository via ZIP and replaces all files safely"""
    exclude_items = {
        'configs',
        'outputs',
        'datasets',
        '.git',
        'config.json',
        'paths.json',
        'orakul_env'
    }  # Protected items that will not be overwritten

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, 'update.zip')

        with console.status(
            '[white]Downloading latest repository archive from GitHub...[/white]',
            spinner='dots',
        ):
            try:
                urllib.request.urlretrieve(REPO_ZIP_URL, zip_path)
            except Exception as e:
                console.print(f'[white][ERROR] Download failed: {e}[/white]')
                return False

        with console.status(
            '[white]Unpacking and preparing core files...[/white]',
            spinner='dots',
        ):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            except Exception as e:
                console.print(f'[white][ERROR] Unpacking failed: {e}[/white]')
                return False

        extracted_folders = [
            f
            for f in os.listdir(temp_dir)
            if os.path.isdir(os.path.join(temp_dir, f)) and f != '__MACOSX'
        ]
        if not extracted_folders:
            console.print('[white][ERROR] Invalid archive structure.[/white]')
            return False

        source_dir = os.path.join(
            temp_dir, extracted_folders[0], 'Orakul_Studio_Core'
        )
        if not os.path.exists(source_dir):
            source_dir = os.path.join(temp_dir, extracted_folders[0])

        with console.status(
            '[white]Overwriting repository files...[/white]',
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
                    console.print(f'[white][WARNING] Failed to update {item}: {e}[/white]')

    return True


def print_banner():
    console.print(r""" [white]                                
 ░█▀█░█▀▄░█▀█░█░█░█░█░█░░░░░█▀▀░▀█▀░█░█░█▀▄░▀█▀░█▀█
 ░█░█░█▀▄░█▀█░█▀▄░█░█░█░░░░░▀▀█░░█░░█░█░█░█░█░█░█
 ░▀▀▀░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░░░▀▀▀░░▀░░▀▀▀░▀▀░░▀▀▀░▀▀▀                                                                                                                                                                                                              
    """)


def run_update():
    os.system('cls' if os.name == 'nt' else 'clear')
    print_banner()
    
    console.print('[white]ORAKUL CORE — Direct GitHub Sync 27.10.2026[/white]')
    console.print('=' * 55)
    
    answer = input('Update/Overwrites all core files from GitHub main branch? (y/n): ').strip().lower()
    if answer == 'y':
        console.print('\n[white]Starting full repository download...[/white]')
        success = download_and_extract_full_repo()
        if success:
            console.print('\n[white][✓] System successfully updated from GitHub![/white]')
        else:
            console.print('\n[white][ERROR] Update failed. Check connection.[/white]')
    else:
        console.print('\n[white]Update cancelled.[/white]')


if __name__ == '__main__':
    run_update()