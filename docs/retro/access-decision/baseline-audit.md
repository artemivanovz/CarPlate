# Baseline audit: access-decision

Снято до правки политики. Код в этот момент не менялся.

## Что есть

Решение собирает `decide()` в `anprgate/access.py`: номер, цвет, кузов. Журнал пишет `Pipeline._finalize` в SQLite параметризованным INSERT. Пароль RTSP режет `mask_credentials` (`video.py:24–26`); в журнал попадает `source.display_name` (`app.py:52`). В `data/anpr.sqlite3` 86 событий, ни у одного `source` не содержит `@`.

`anpr.yaml` на посту: `fuzzy_decision: review`, `color_check: false`, `body_check: true`, `body_match: 0.75`. Цвет в этом файле не участвует. Дыра «кузов отменяет цвет» на этом файле спит и проявляется, когда `color_check` включают (это значение по умолчанию в `AccessConfig`, `config.py:92`).

Тесты до новых кейсов: `tests/test_core.py`, 23 проходящих после добавления матрицы без кузова. Git-репозитория нет, истории коммитов нет. `ruff` и `pytest-cov` в `.venv` не установлены.

## Находки до правки

| Находка | Указатель |
|---------|-----------|
| `body.score >= body_match` ставил `decision = ALLOW` и удалял причины, начинающиеся с «цвет » | `access.py` `_check_body`, до правки |
| `escalate(cfg.fuzzy_decision)` уважал `allow` | `_check_plate`, до правки |
| `body_embedder is None` возвращал «нет эталона кузова» | `pipeline.py` `_body_match`, до правки |
| Чужая размерность вектора возвращала «нет кадра кузова» | там же |
| `except Exception` логировал стек, но при `color_check: true` причина не попадала в `reasons`, решение оставалось `allow` | `pipeline.py` except + `_check_body` ветка skip |
| SQL с плейсхолдерами `?`, методы `Database` под `RLock`, соединение с `check_same_thread=False` | `db.py:115` |
| Кнопки карточки без `objectName` / accessibleName | `event_popup.py` |

TODO/FIXME внутри периметра нет.
