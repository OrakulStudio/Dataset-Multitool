@echo off
chcp 65001 > nul
echo =================================================================
echo       ORAKUL STUDIO - AUTO SETUP / АВТОМАТИЧЕСКАЯ УСТАНОВКА
echo =================================================================
echo.
echo [1/5] Установка базовых библиотек / Installing base libraries...
python -m pip install -r requirements.txt

echo.
echo [2/5] Создание виртуального окружения / Creating orakul_env...
python -m venv orakul_env

echo.
echo [3/5] Подготовка окружения (Pre-installing dependencies)...
orakul_env\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
orakul_env\Scripts\python.exe -m pip install jinja2

echo.
echo [4/5] Установка PyTorch CUDA (~2.5GB)...
echo Пожалуйста, подождите, идет скачивание Torch...
orakul_env\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo.
echo [5/5] Установка остальных нейромодулей (Transformers, Qwen, Rich)...
orakul_env\Scripts\python.exe -m pip install -r requirements_caption_env.txt

echo.
echo =================================================================
echo [+] Setup completed successfully! / Установка успешно завершена!
echo [+] You can now run Orakul Studio / Теперь можно запускать меню.
echo =================================================================
echo Press any key to exit / Нажмите любую клавишу для выхода...
pause > nul