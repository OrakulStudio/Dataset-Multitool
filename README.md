# 🛠️ Orakul Toolkit: Viking Multi-tool

**A portable multi-tool for preparing datasets and managing diffusion model weights.**
Built entirely in Python; no installation required simply unzip and run. The interface switches between **Russian and English on the fly**, right while the program is running.

Developed and maintained by [Orakul Studio](https://github.com/OrakulStudio) (Chernihiv, Ukraine 🇺🇦).

---

## About the Project

This is not just a single script, but a comprehensive pipeline of 9 tools covering the entire process from raw images to fine-tuning and model weight merging. Each module can be used independently or as part of an end-to-end workflow.

---

## 🧩 Module Overview (9 Powerful Tools)

The pipeline is divided into logical stages, ranging from raw images to weight fine-tuning:

### 1. Viking Caption
Multimodal auto captioning powered by state-of-the-art models (e.g., `Qwen`), featuring system prompt support via a built-in editor. Supports convenient drag-and-drop of folders directly into the terminal.

### 2. Lore Description Generator (`Qwen Poem`)
Analyzes your image gallery to create artistic, aesthetic descriptions for releases on platforms like Civitai or Hugging Face.

### 3. Batch Cropping (`Crop & Rename`)
Precise image cropping into perfect squares (ranging from 1024px to 3000px+) with sequential file renaming and no quality loss.

### 4. Dataset Manager (`Dataset Manager`)
Augmentation tools: instant image mirroring (left-to-right) with automatic filename recalculation. 

### 5. Text Scanner (`QC Scanner`)
Statistical analysis of thousands of text files: detecting anomalies, text looping, and drastic deviations in description length.

### 6. Surgical Cleanup (`Auto Clean`)
Automatic removal of statistical "noise" and text artifacts based on scanner reports.

### 7. Architecture Inspector (`Inspector`)
Lightning-fast reading of headers from massive `safetensors` files (~40 GB) in 0.9 seconds, without fully loading weights into RAM.

### 8. Metadata Injector (`Metadata Injector`)
Restoring and carefully embedding JSON generation recipes (prompts, samplers) into compressed Photoshop images for correct reading on Civitai.

### 9. Weight Merger Hub (`Weight Merger Hub`)
Direct weight injection, true rank concatenation, and advanced SVD distillation technology (Matrix Compression). Enables compressing hybrid models and merging 160 Flux.2 layers in seconds.

---

## 🌍 Bilingual Interface

Full **RU / EN** support with on-the-fly switching no program restart required; the same build works for both Russian and English-speaking users.

---
[![Orakul Studio Core Demo](https://markdown-videos-api.jorgenkh.no/youtube/NOOhGBC6DP0)](https://youtu.be/NOOhGBC6DP0)
---

## 🚀 Installation

The virtual environment is **not included** in the distribution neither on GitHub nor on Hugging Face. The reason is simple: a `venv` stores a hard link to the Python interpreter of the machine where it was created; consequently, it fails to work when transferred to another PC (this is a characteristic of Python itself, not a limitation of the toolkit). However, this gives you complete freedom: you can place `orakul_env` and the Qwen model cache (~4 GB) on any drive you choose.

### Option A: Single command setup (Windows, recommended)

```bash
setup.bat
```

The script will automatically install lightweight dependencies, create `orakul_env`, install `torch` for CUDA 12.4, and install all the heavy Qwen libraries in the correct order. Once finished, run `run.bat` or `python run.py`.

### Option B: Manual setup (Windows/Linux, if you need control over the CUDA version)

```bash
cd Orakul_Studio_Core

# 1. Lightweight dependencies for the menu itself (run.py)
pip install -r requirements.txt

# 2. Create a venv for heavy Qwen/Weight Merger tasks
python -m venv orakul_env
orakul_env\Scripts\activate          # Linux/Git Bash: source orakul_env/Scripts/activate
pip install -r requirements_caption_env.txt
deactivate

# 3. Launch
python run.py
```

> 💡 If your GPU is not an RTX 4090, install `torch` separately for your specific CUDA version before step 2 (generate the command at [pytorch.org](https://pytorch.org/get-started/locally/)), and only then run `pip install -r requirements_caption_env.txt`; pip will not overwrite a suitable version that is already installed.

If `run.py` is launched without `orakul_env`, it will display this same sequence of commands directly in the terminal (in both Russian and English) and guide you on what to do. ### Initial Launch of Qwen Modules

Upon the first launch of **Viking Caption** or **Qwen Poem**, the utility will prompt you to select a drive letter for the model cache (~4 GB); it intentionally avoids selecting a default drive automatically to prevent using storage space where it wasn't requested. Your selection is saved in `paths.json` and reused for subsequent launches on the same machine.

---

## 📋 Requirements

- Python 3.10+
- CUDA enabled GPU (recommended for Qwen captioning and weight merging)
- Sufficient disk space for temporary files when working with safetensors (models ~40 GB+) and for the Qwen cache (~4 GB)

---

## 📜 License

*MIT*

## 🔗 Links

- [Hugging Face](https://huggingface.co/OrakulStorm)
- [CivitAI](https://civitai.com/user/ORAKUL_STUDIO)
