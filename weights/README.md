# Веса моделей ANPR Gate

Веса хранятся отдельно от Git. Git LFS не используется. Существующие файлы на
машине оператора сохраняются; правила исключения не удаляют их с диска.

Для обычной конфигурации нужны:

```text
weights/yolo26n.pt
weights/ru_plate/plate_detector.pt
weights/ru_plate/plate_recognizer.pt
```

Для необязательного `models.body.backend: fastreid` дополнительно нужен
`weights/reid/veri_sbs_R50-ibn.pth`. Backend `shape` не требует своих весов.

После клонирования перенесите эти файлы из проверенной поставки, сохраняя
структуру каталогов. Для Windows в PowerShell:

```powershell
Copy-Item -Path 'D:\CarPlate-assets\weights\*' -Destination '.\weights' -Recurse
.\.venv\Scripts\python.exe scripts\check_assets.py
```

Для Linux:

```bash
cp -r /mnt/CarPlate-assets/weights/. ./weights/
.venv/bin/python scripts/check_assets.py
```

Пути `D:\CarPlate-assets` и `/mnt/CarPlate-assets` замените на каталог вашей
поставки. `manifest.json` фиксирует размеры и SHA256 весов, с которыми проверялся
проект. Проверка не загружает и не заменяет файлы. Флаг `--all` дополнительно
требует FastReID. При осознанной замене моделей обновите манифест и проверки.

Происхождение русского OCR описано в
[SOURCE.md](../third_party/ru_plate_anpr/SOURCE.md). При отсутствии YOLO26n
Ultralytics может скачать веса по имени; для воспроизводимого запуска
восстанавливайте проверенную поставку целиком.
