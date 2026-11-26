# Происхождение

- Код `ocr_model/` и `full_stand/` скопирован без изменений из
  https://github.com/crinzon/rover-anpr (ветка main, 30.10.2026), который, в свою очередь,
  использует модели https://github.com/shershunov/russian-license-plate-anpr.
- Веса (`weights/ru_plate/*.pt`) загружены из shershunov/russian-license-plate-anpr,
  ревизия `faa895e481b6538783184ad50d304aa453b4e794`; SHA256 совпадает с `download_models.py` rover-anpr:
  - plate_detector.pt — `0ee9b29633e38d1ca444124791036665bbb28a563e3cdae765a72c7fd3ea1312`
  - plate_recognizer.pt — `b115566f6894da055780959d1497f10ad0fd8b461a2ef533d08386cf135f6ac4`

## Лицензия

В обоих репозиториях файл лицензии отсутствует. По умолчанию это означает, что права не переданы.
Для пилота и оценки — допустимо на усмотрение руководителя; для внедрения нужно письменное
разрешение автора либо переобучение моделей на собственных данных.
Детектор основан на Ultralytics YOLO26n (AGPL-3.0).

Для старых окружений с torch 2.2 в `anprgate/models/ru_plate_reader.py` сохранена замена `F.rms_norm`.
Стандартное окружение использует torch 2.14.1 с нативной реализацией этой операции.

Локальная правка безопасности: в `ocr_model/nn/inference.py` загрузка checkpoint
переведена на `torch.load(..., weights_only=True)` без небезопасного fallback.
