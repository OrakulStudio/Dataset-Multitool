# ==============================================================================
#  ORAKUL CORE — AI Dataset & Model MultiTool
#  Module: Metadata Injector
# ------------------------------------------------------------------------------
#  Author:      Orakul (Orakul Studio)
#  GitHub:      https://github.com/OrakulStudio
#  Civitai:     https://civitai.com/user/ORAKUL_STUDIO
#  HuggingFace: https://huggingface.co/OrakulStorm
# ------------------------------------------------------------------------------
#  Description: Transfer prompts and workflows from ComfyUI PNGs to retouched images.
#  Copyright (c) 2026 Orakul Studio. All rights reserved.
# ==============================================================================

import json
import os
from pathlib import Path
import struct
import sys
from rich.console import Console
from rich.panel import Panel
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


def extract_text_chunks_from_png(png_bytes):
    """Extract all text metadata chunks (tEXt, iTXt, zTXt) from PNG"""
    if not png_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('File is not a valid PNG!')

    chunks = []
    idx = 8

    while idx < len(png_bytes):
        length = struct.unpack('>I', png_bytes[idx : idx + 4])[0]
        chunk_type = png_bytes[idx + 4 : idx + 8]
        chunk_full = png_bytes[idx : idx + 12 + length]

        if chunk_type in (b'tEXt', b'iTXt', b'zTXt'):
            chunks.append((chunk_type, chunk_full))

        idx += 12 + length
        if chunk_type == b'IEND':
            break

    return chunks


def inspect_png_metadata(png_path: Path):
    """Inspect and optionally export metadata from a PNG file"""
    if not png_path.exists():
        console.print(Panel(
            f"[bold white]\\[ERROR][/bold white] [dim]File not found: {png_path}[/dim]",
            border_style="grey50",
            box=box.ROUNDED
        ))
        return

    try:
        with open(png_path, 'rb') as f:
            b = f.read()
        extracted = extract_text_chunks_from_png(b)

        if not extracted:
            console.print(Panel(
                f"[dim]No text metadata found in file: {png_path.name}[/dim]",
                border_style="grey50",
                box=box.ROUNDED
            ))
            return

        info_lines = [f"[white]Text chunks found:[/white] [dim]{len(extracted)}[/dim]"]
        parsed_data = {}

        for item in extracted:
            c_type, c_data = item
            try:
                text_content = c_data[8:-4].decode('utf-8', errors='ignore')
                parts = text_content.split('\x00', 1)
                key = parts[0]
                val = parts[1] if len(parts) > 1 else ''
                parsed_data[key] = val

                preview = val[:120].replace('\n', ' ') if val else ''
                info_lines.append(f"  [white]• Key:[/white] [dim]{key}[/dim]")
                info_lines.append(f"    [dim]Preview: {preview}...[/dim]")
            except Exception:
                pass

        console.print(Panel(
            '\n'.join(info_lines),
            title=f"[bold white] METADATA ({png_path.name}) [/bold white]",
            border_style="grey50",
            box=box.ROUNDED
        ))

        export_choice = console.input(
            "\n[bold white]> Export Prompt (.txt) or Workflow (.json)? (y/n) [Enter = n]: [/bold white]"
        ).strip().lower()

        if export_choice in ('y', 'yes'):
            export_dir = png_path.parent / f'{png_path.stem}_exported'
            export_dir.mkdir(parents=True, exist_ok=True)

            for key, val in parsed_data.items():
                if key == 'prompt':
                    out_file = export_dir / 'prompt.txt'
                    try:
                        p_json = json.loads(val)
                        texts = []
                        for node_id, node_data in p_json.items():
                            if isinstance(node_data, dict) and 'inputs' in node_data:
                                for k, v in node_data['inputs'].items():
                                    if isinstance(v, str) and len(v) > 20:
                                        texts.append(v)
                        raw_text = '\n\n'.join(texts) if texts else val
                    except Exception:
                        raw_text = val

                    out_file.write_text(raw_text, encoding='utf-8')
                    console.print(f"[dim]+ Prompt saved:[/dim] {out_file}")

                elif key == 'workflow':
                    out_file = export_dir / 'workflow.json'
                    try:
                        w_json = json.loads(val)
                        out_file.write_text(
                            json.dumps(w_json, indent=2, ensure_ascii=False),
                            encoding='utf-8',
                        )
                    except Exception:
                        out_file.write_text(val, encoding='utf-8')
                    console.print(f"[dim]+ Workflow saved:[/dim] {out_file}")
                else:
                    out_file = export_dir / f'{key}.txt'
                    out_file.write_text(val, encoding='utf-8')
                    console.print(f"[dim]+ Key '{key}' saved:[/dim] {out_file}")

            console.print(Panel(
                f"[white]All files exported to:[/white] [dim]{export_dir}[/dim]",
                border_style="grey50",
                box=box.ROUNDED
            ))

    except Exception as e:
        console.print(Panel(
            f"[bold white]\\[ERROR][/bold white] [dim]Failed to parse file: {e}[/dim]",
            border_style="grey50",
            box=box.ROUNDED
        ))


def transfer_png_metadata(src_png_path: Path, tgt_png_path: Path, out_png_path: Path):
    if not src_png_path.exists():
        console.print(Panel(
            f"[bold white]\\[ERROR][/bold white] [dim]Original PNG not found: {src_png_path}[/dim]",
            border_style="grey50",
            box=box.ROUNDED
        ))
        return False
    if not tgt_png_path.exists():
        console.print(Panel(
            f"[bold white]\\[ERROR][/bold white] [dim]Processed PNG not found: {tgt_png_path}[/dim]",
            border_style="grey50",
            box=box.ROUNDED
        ))
        return False

    with open(src_png_path, 'rb') as f:
        src_bytes = f.read()

    try:
        extracted = extract_text_chunks_from_png(src_bytes)
        meta_chunks = [item[1] for item in extracted]
    except Exception as e:
        console.print(Panel(
            f"[bold white]\\[ERROR][/bold white] [dim]Metadata read error: {e}[/dim]",
            border_style="grey50",
            box=box.ROUNDED
        ))
        return False

    if not meta_chunks:
        console.print(Panel(
            "[dim]Warning: No text metadata chunks found in original PNG[/dim]",
            border_style="grey50",
            box=box.ROUNDED
        ))
    else:
        console.print(Panel(
            f"[white]Extracted metadata chunks:[/white] [dim]{len(meta_chunks)}[/dim]",
            border_style="grey50",
            box=box.ROUNDED
        ))

    with open(tgt_png_path, 'rb') as f:
        tgt_bytes = f.read()

    if not tgt_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        console.print(Panel(
            "[bold white]\\[ERROR][/bold white] [dim]Processed file is not a valid PNG[/dim]",
            border_style="grey50",
            box=box.ROUNDED
        ))
        return False

    ihdr_len = struct.unpack('>I', tgt_bytes[8:12])[0]
    ihdr_end = 8 + 4 + 4 + ihdr_len + 4

    final_bytes = (
        tgt_bytes[:ihdr_end] + b''.join(meta_chunks) + tgt_bytes[ihdr_end:]
    )

    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_png_path, 'wb') as f:
        f.write(final_bytes)

    console.print(Panel(
        f"[white]SUCCESS[/white]\n[dim]Saved to: {out_png_path}[/dim]",
        border_style="grey50",
        box=box.ROUNDED
    ))
    return True


def main():
    console.print(Panel(
        "[dim]Transfer prompts and workflows from ComfyUI PNGs into retouched images[/dim]",
        title="[bold white] METADATA INJECTOR [/bold white]",
        border_style="grey50",
        box=box.ROUNDED
    ))

    while True:
        menu_text = (
            "[bold white][ 1 ][/bold white] Transfer metadata (Original → Retouch pipeline)\n"
            "[bold white][ 2 ][/bold white] Inspect / export PNG metadata\n"
            "[bold white][ 0 ][/bold white] Return to main menu"
        )
        console.print(Panel(
            menu_text,
            title="[bold white] MODULE MENU [/bold white]",
            border_style="grey50",
            box=box.ROUNDED
        ))

        mode = console.input("\n[bold white]> Select action (0-2): [/bold white]").strip()

        if mode in ('0', ''):
            console.print("[dim]Returning to main menu...[/dim]")
            break

        # Mode 1: Continuous transfer pipeline
        if mode == '1':
            console.print(Panel(
                "[white]CONTINUOUS PIPELINE[/white] [dim](Original → Retouch)[/dim]",
                border_style="grey50",
                box=box.ROUNDED
            ))
            
            folder_input = console.input(
                "[bold white]> Output folder [Enter = auto 'published']: [/bold white]"
            ).strip(' "\'')

            if folder_input:
                dest_folder = Path(folder_input).resolve()
            else:
                dest_folder = Path.cwd() / 'published'
                
            dest_folder.mkdir(parents=True, exist_ok=True)
            console.print(Panel(
                f"[white]Working folder:[/white] [dim]{dest_folder}[/dim]\n"
                "[dim]Enter '0' instead of a path to exit[/dim]",
                border_style="grey50",
                box=box.ROUNDED
            ))

            pair_counter = 1
            while True:
                console.print(f"\n[white]── Pair #{pair_counter} ──[/white]")
                orig_input = console.input(
                    "[bold white]> 1. Drag original ComfyUI PNG (or \"0\" to cancel): [/bold white]"
                ).strip(' "\'')
                if orig_input == '0' or not orig_input:
                    console.print("[dim]Returning to module menu...[/dim]")
                    break
                orig_path = Path(orig_input).resolve()

                proc_input = console.input(
                    "[bold white]> 2. Drag processed/retouched PNG (or \"0\" to cancel): [/bold white]"
                ).strip(' "\'')
                if proc_input == '0' or not proc_input:
                    console.print("[dim]Returning to module menu...[/dim]")
                    break
                proc_path = Path(proc_input).resolve()

                out_path = dest_folder / f'{proc_path.stem}_meta.png'

                success = transfer_png_metadata(orig_path, proc_path, out_path)
                
                if success:
                    pair_counter += 1

        # Mode 2: Inspect / Export
        elif mode == '2':
            file_to_check = console.input(
                "\n[bold white]> Drag PNG for inspection (or \"0\" to cancel): [/bold white]"
            ).strip(' "\'')
            if file_to_check and file_to_check != '0':
                inspect_png_metadata(Path(file_to_check).resolve())


if __name__ == '__main__':
    main()