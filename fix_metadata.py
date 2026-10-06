#!/usr/bin/env python3
"""
Модуль восстановления метаданных (Metadata Injector) для Orakul Studio.
Поддерживает экспорт промптов (.txt) и воркфлоу (.json) в один клик.
"""

import json
import os
from pathlib import Path
import struct
import sys
from rich.console import Console
from rich.panel import Panel

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
# -----------------------------------------

console = Console()


def extract_text_chunks_from_png(png_bytes):
  """Вытаскивает все текстовые метаданные (tEXt, iTXt, zTXt) из PNG"""
  if not png_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
    raise ValueError('Файл не является валидным PNG!')

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
  """Инструмент проверки и экспорта метаданных из файла"""
  if not png_path.exists():
    console.print(t("[bold red][-] Файл не найден:[/bold red]") + f" [bold red]{png_path}[/bold red]")
    return

  try:
    with open(png_path, 'rb') as f:
      b = f.read()
    extracted = extract_text_chunks_from_png(b)

    if not extracted:
      console.print(
          t("[bold yellow][!] В файле[/bold yellow]") + f" [bold yellow]'{png_path.name}'[/bold yellow] " +
          t("[bold yellow]НЕ НАЙДЕНО текстовых метаданных.[/bold yellow]")
      )
      return

    info_lines = [
        t("[bold green]✓ Найдено текстовых чанков:[/bold green]") + f" [bold green]{len(extracted)}[/bold green]"
    ]
    parsed_data = {}

    for item in extracted:
      c_type, c_data = item
      try:
        text_content = c_data[8:-4].decode('utf-8', errors='ignore')
        parts = text_content.split('\x00', 1)
        key = parts[0]
        val = parts[1] if len(parts) > 1 else ''
        parsed_data[key] = val

        preview = val[:150].replace('\n', ' ') if val else ''
        info_lines.append(t("  • Ключ: ") + f"[bold white]{key}[/bold white]")
        info_lines.append(t("    [dim]Превью:[/dim]") + f" [dim]{preview}...[/dim]")
      except Exception:
        pass

    console.print(
        Panel(
            '\n'.join(info_lines),
            title=t("[bold green]🔍 МЕТАДАННЫЕ ФАЙЛА[/bold green]") + f" [bold green]({png_path.name})[/bold green]",
            border_style='green',
            expand=False,
        )
    )

    # Предлагаем экспортировать в файлы
    export_choice = (
        console.input(
            '\n' + t("[bold yellow]Хочешь выгрузить Prompt (.txt) или Workflow (.json) из этого файла? (y/n) [Enter = n]: [/bold yellow]")
        )
        .strip()
        .lower()
    )
    if export_choice in ('y', 'yes', 'д', 'да'):
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
          except:
            raw_text = val

          out_file.write_text(raw_text, encoding='utf-8')
          console.print(
              t("[bold green][+] Промпт сохранён:[/bold green]") + f" {out_file}"
          )

        elif key == 'workflow':
          out_file = export_dir / 'workflow.json'
          try:
            w_json = json.loads(val)
            out_file.write_text(
                json.dumps(w_json, indent=2, ensure_ascii=False),
                encoding='utf-8',
            )
          except:
            out_file.write_text(val, encoding='utf-8')
          console.print(
              t("[bold green][+] Воркфлоу сохранён:[/bold green]") + f" {out_file}"
          )
        else:
          out_file = export_dir / f'{key}.txt'
          out_file.write_text(val, encoding='utf-8')
          console.print(
              t("[bold green][+] Ключ[/bold green]") + f" '{key}' " + t("[bold green]сохранён:[/bold green]") + f" {out_file}"
          )

      console.print(
          t("[cyan][*] Все файлы выгружены в папку:[/cyan]") + f" {export_dir}"
      )

  except Exception as e:
    console.print(t("[bold red][-] Ошибка при разборе файла:[/bold red]") + f" {e}")


def transfer_png_metadata(
    src_png_path: Path, tgt_png_path: Path, out_png_path: Path
):
  if not src_png_path.exists():
    console.print(
        t("[bold red][-] Ошибка: Оригинальный PNG не найден ->[/bold red]") + f" [bold red]{src_png_path}[/bold red]"
    )
    return False
  if not tgt_png_path.exists():
    console.print(
        t("[bold red][-] Ошибка: Обработанный PNG не найден ->[/bold red]") + f" [bold red]{tgt_png_path}[/bold red]"
    )
    return False

  with open(src_png_path, 'rb') as f:
    src_bytes = f.read()

  try:
    extracted = extract_text_chunks_from_png(src_bytes)
    meta_chunks = [item[1] for item in extracted]
  except Exception as e:
    console.print(t("[bold red][-] Ошибка чтения метаданных:[/bold red]") + f" {e}")
    return False

  if not meta_chunks:
    console.print(
        t("[bold yellow][!] Внимание: В оригинальном PNG не найдено текстовых чанков метаданных.[/bold yellow]")
    )
  else:
    console.print(
        t("[bold cyan][*] Извлечено чанков метаданных:[/bold cyan]") + f" {len(meta_chunks)}"
    )

  with open(tgt_png_path, 'rb') as f:
    tgt_bytes = f.read()

  if not tgt_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
    console.print(
        t("[bold red][-] Ошибка: Обработанный файл не является валидным PNG.[/bold red]")
    )
    return False

  ihdr_len = struct.unpack('>I', tgt_bytes[8:12])[0]
  ihdr_end = 8 + 4 + 4 + ihdr_len + 4

  final_bytes = (
      tgt_bytes[:ihdr_end] + b''.join(meta_chunks) + tgt_bytes[ihdr_end:]
  )

  out_png_path.parent.mkdir(parents=True, exist_ok=True)
  with open(out_png_path, 'wb') as f:
    f.write(final_bytes)

  console.print(
      t("[bold green][+] УСПЕШНО! Файл сохранён:[/bold green]") + f"\n    [cyan]{out_png_path}[/cyan]"
  )
  return True


def main():
  console.print(
      Panel(
          t("[bold cyan]Инструмент переноса промптов и воркфлоу из ComfyUI в ретушированные кадры[/bold cyan]"),
          title=t("[bold yellow]⚡ METADATA INJECTOR ⚡[/bold yellow]"),
          border_style='yellow',
          expand=False,
      )
  )

  while True:
    console.print('\n' + t("[bold white]────────── МЕНЮ МОДУЛЯ ──────────[/bold white]"))
    console.print(
        t("[bold green][1][/bold green] Перенести метаданные (Оригинал ➔ Ретушь)")
    )
    console.print(
        t("[bold green][2][/bold green] Проверить / экспортировать метаданные PNG")
    )
    console.print(t("[bold red][0][/bold red] Выход в главное меню"))

    mode = console.input('\n' + t("[bold yellow]Выберите действие (0-2): [/bold yellow]")).strip()

    if mode in ('0', ''):
      console.print(t("[dim]Возврат в главное меню...[/dim]"))
      break

    # === РЕЖИМ 1: ПЕРЕНОС МЕТАДАННЫХ ===
    if mode == '1':
      print('\n' + '=' * 50)
      orig_input = (
          console.input(
              t("[bold cyan]1. Перетащи оригинальный PNG из ComfyUI (или \"0\" для отмены):[/bold cyan] ")
          )
          .strip(' "\'')
      )
      if orig_input == '0' or not orig_input:
        continue
      orig_path = Path(orig_input).resolve()

      proc_input = (
          console.input(
              t("[bold cyan]2. Перетащи обработанный PNG после ретуши (или \"0\" для отмены):[/bold cyan] ")
          )
          .strip(' "\'')
      )
      if proc_input == '0' or not proc_input:
        continue
      proc_path = Path(proc_input).resolve()

      default_folder = proc_path.parent / 'published'
      folder_input = (
          console.input(
              t("[bold yellow]3. Папка для сохранения [Enter = [/bold yellow]") +
              f"[bold yellow]'{default_folder.name}']: [/bold yellow]"
          )
          .strip(' "\'')
      )

      if folder_input:
        dest_folder = Path(folder_input).resolve()
      else:
        dest_folder = default_folder

      out_path = dest_folder / f'{proc_path.stem}_meta.png'

      success = transfer_png_metadata(orig_path, proc_path, out_path)

      if success:
        check_choice = (
            console.input(
                '\n' + t("[bold yellow]Хочешь проверить метаданные готового файла? (y/n) [Enter = n]: [/bold yellow]")
            )
            .strip()
            .lower()
        )
        if check_choice in ('y', 'yes', 'д', 'да'):
          inspect_png_metadata(out_path)

      # Полный сброс переменных перед новым кругом
      orig_path = None
      proc_path = None
      out_path = None
      print('=' * 50)

    # === РЕЖИМ 2: ПРОВЕРКА / ЭКСПОРТ ===
    elif mode == '2':
      file_to_check = (
          console.input(
              '\n' + t("[bold cyan]Перетащи PNG для инспекции (или \"0\" для отмены):[/bold cyan] ")
          )
          .strip(' "\'')
      )
      if file_to_check and file_to_check != '0':
        inspect_png_metadata(Path(file_to_check).resolve())


if __name__ == '__main__':
  main()