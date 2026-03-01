#!/bin/bash
# Настройка окружения на удалённом сервере после клонирования репозитория.
# Запуск: из корня проекта: bash scripts/setup_server.sh

set -e
cd "$(dirname "$0")/.."
echo "Рабочая директория: $(pwd)"

# Python 3.10+ желателен (для проекта использовался 3.11)
if ! command -v python3 &>/dev/null; then
  echo "Установите python3 (например: sudo apt install python3 python3-venv python3-pip)"
  exit 1
fi

# Виртуальное окружение
if [ ! -d "venv" ]; then
  python3 -m venv venv
  echo "Создано виртуальное окружение venv"
fi
source venv/bin/activate

# Зависимости
pip install --upgrade pip
pip install -r requirements.txt
echo "Зависимости установлены."

# Конфиг: копируем дефолтный, если нет своего
if [ ! -f "config_local.yaml" ]; then
  cp config_default.yaml config_local.yaml
  echo "Создан config_local.yaml — при необходимости отредактируйте (API ключи, device: cuda/cpu)."
fi

echo ""
echo "Готово. Запуск пайплайна (CLI):"
echo "  source venv/bin/activate"
echo "  python run_pipeline.py --video путь/к/видео.mp4 --task 'описание сценария' --output результат.mp4"
echo "  python run_pipeline.py --video input.mp4 --task-file situations.txt --output out.mp4"
echo ""
echo "Веб-интерфейс (доступ с других машин: host=0.0.0.0):"
echo "  source venv/bin/activate"
echo "  python web/app.py"
echo "  # или: flask --app web.app run --host 0.0.0.0 --port 5000"
echo "  # затем откройте в браузере: http://адрес_сервера:5000"
