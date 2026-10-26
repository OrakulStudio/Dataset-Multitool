# ==============================================================================
#  ORAKUL CORE — AI Dataset & Model MultiTool
#  Module: Auto Updater (GitHub Zip-Sync)
# ------------------------------------------------------------------------------
#  Author:      Orakul (Orakul Studio)
#  GitHub:      https://github.com/OrakulStudio
#  Civitai:     https://civitai.com/user/ORAKUL_STUDIO
#  HuggingFace: https://huggingface.co/OrakulStorm
# ------------------------------------------------------------------------------
#  Description: High-performance background updater pulling full repo via zip.
#  Copyright (c) 2026 Orakul Studio. All rights reserved.
# ==============================================================================

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
from rich import box

# Force UTF-8 encoding for Windows console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

console = Console()

GITHUB_USER = 'OrakulStudio'
REPO_NAME = 'Dataset-Multitool'
REPO_ZIP_URL = f'https://github.com/{GITHUB_USER}/{REPO_NAME}/archive/refs/heads/main.zip'


def get_local_commit_hash():
    """Get the local git commit hash"""
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
    """Get the latest commit hash from GitHub API (timeout 2 sec)"""
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
    """Downloads entire repository via ZIP and updates files safely"""
    exclude_items = {
        'configs',
        'outputs',
        'datasets',
        '.git',
        'config.json',
        'paths.json'
    }  # Items that should not be overwritten

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, 'update.zip')

        with console.status(
            '[white]Downloading repository archive from GitHub...[/white]',
            spinner='dots',
        ):
            try:
                urllib.request.urlretrieve(REPO_ZIP_URL, zip_path)
            except Exception as e:
                console.print(f'[white][ERROR] Download failed: {e}[/white]')
                return False

        with console.status(
            '[white]Unpacking and preparing files...[/white]',
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
            '[white]Safely updating files...[/white]',
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


def draw_whats_new():
    changelog = (
        '[white]INFO: WHAT\'S NEW ON GITHUB[/white]\n\n'
        '  [white]— Qwen-VL Engine —[/white]\n'
        '  [dim]Support for 3 models (2.5-VL-7B, 3-VL-8B, 3-VL-30B-FP8).[/dim]\n'
        '  [dim]Qwen 2.5-VL recommended for standard workflows.[/dim]\n\n'
        '  [white]— Crop & Rename —[/white]\n'
        '  [dim]Optional prefix support & smart sync for paired .txt files.[/dim]\n\n'
        '  [white]— Z-Image SAEP-SVD —[/white]\n'
        '  [dim]Custom distillation algorithm with zero microcontrast loss.[/dim]\n\n'
        '  [white]— UI & Optimization —[/white]\n'
        '  [dim]Unified monochrome TUI core layout across all modules.[/dim]'
    )

    console.print(
        Panel(
            changelog,
            title='[white] ORAKUL DATASET MULTITOOL [/white]',
            title_align='left',
            subtitle='[white]GitHub Sync[/white]',
            subtitle_align='right',
            border_style='dim',
            box=box.ROUNDED,
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
        os.system('cls' if os.name == 'nt' else 'clear')
        print_banner()
        draw_whats_new()
        console.print(
            f'\n[white]New version available on GitHub! (Commit: {remote_hash})[/white]'
        )
        console.print('=' * 55)

        answer = (
            input('Update Dataset-Multitool right now? (y/n): ')
            .strip()
            .lower()
        )
        if answer == 'y':
            console.print('\n[white]Starting full repository synchronization...[/white]')
            success = download_and_extract_full_repo()
            if success:
                console.print('[white]Successfully updated! Please restart the tool.[/white]')
                sys.exit(0)
            else:
                console.print('[white][ERROR] Failed to update files.[/white]')


if __name__ == '__main__':
    check_for_updates()