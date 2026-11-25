# Gap register: access-decision

Current phase: 5
Perimeter: `anprgate/access.py`, `anprgate/pipeline.py` (`_finalize`, `_body_match`), `anprgate/db.py`, `anprgate/video.py`, `anprgate/ui/event_popup.py`, `anprgate/ui/main_window.py` (`_on_operator_action`), `anpr.yaml` (`access:`), `tests/test_core.py`, `README.md` (decision diagram).
Entry point: `decide()` then `Pipeline._finalize`.
Callers: `Pipeline._finalize`, `tests/test_core.py`.
Blast radius: journal `events.decision` / `events.reasons`, operator card, overlay status. Not the plate OCR models.
Out of slice: YOLO weights, `anprgate eval` accuracy table, vehicle dialog layout.

| ID | Axis | Gap | Evidence | Forward fix | Severity | Effort | Status |
|----|------|-----|----------|-------------|----------|--------|--------|
| G-001 | practice | Кузов с близостью ≥ `body_match` переписывал `deny` по цвету в `allow` и стирал причины цвета | было `anprgate/access.py` `_check_body`; тест `test_body_match_does_not_override_color_deny` | Отказ по цвету сохраняется, строка 129–133 | P0 | S | done |
| G-002 | practice | `fuzzy_decision: allow` давал проезд при одном отличии символа; `deny` прятал сомнение | `anprgate/access.py` `_check_plate`; тесты `test_fuzzy_config_*` | Всегда `review`, строки 171–175 | P0 | S | done |
| G-003 | practice | Сбой сверки маскировался («нет эталона» / «нет кадра») и при включённом цвете оставался `allow` без записи в `reasons` | `pipeline.py` 296–315, `access.py` 119–125 | Причина «не выполнена» и `review` | P0 | S | done |
| G-004 | practice | Нет кадра кузова (`pipeline.py:306`) при `color_check: true` по-прежнему этап `skip` и решение может остаться `allow`; в журнал этапов нет, только `reasons` | `access.py:125`, `db.py` INSERT events | Отдельное решение: нет кадра = `review` даже днём | P1 | S | open |
| G-005 | missing | Нет входа оператора. «Пропустить вручную» доступно любому у окна | `event_popup.py:115–119` | Локальная роль оператора до записи `operator_action` | P0 | M | open |
| G-006 | missing | Действие оператора не меняет `events.decision` и затирается повторным открытием карточки | `db.py:247–251`, `main_window.py:214–217` | История действий, баннер не расходится с колонкой решения | P1 | M | open |
| G-007 | thinking | Поле `fuzzy_decision` больше не влияет на исход, но остаётся в `AccessConfig` | `config.py:91`, `anpr.yaml:57` | Убрать ключ или отклонять `allow`/`deny` при загрузке | P2 | S | open |
| G-008 | selection | Ultralytics AGPL и `ru_anpr` без лицензии | `README.md` раздел лицензий | Блокер коммерческой выкладки, не бага кадра | P0 | L | open |
| G-009 | missing | Снимки номеров на диске без срока хранения | `data/snapshots/`, `pipeline.py` `_save_snapshots` | Политика хранения — только после решения владельца | P2 | M | open |
| G-010 | practice | Нет git, нет ruff в `.venv`, нет pytest-cov | прогон 2026-11-06 | Поставить инструменты локально, не как часть политики допуска | P2 | S | open |

Не чинить в этом срезе: замена `ru_anpr` на `fast_alpr`, `frame_interval`, пороги `body_match` / `body_alarm`.
