# Acceptance Brief: политика допуска

**Status:** Draft
**Revision:** 1
**Prepared for:** оператор поста ANPR-Gate
**Approval required before risky work:** No — инварианты заданы в запросе, правка только сужает `allow`.

## Revision Log

| Rev | Date | Changed criteria | Reason |
| --- | --- | --- | --- |
| 1 | 2026-11-06 | — | Initial draft |

## Goal

Чужой или сомнительный проезд не получает автоматическое «пропустить».

## Scope

**In scope**

- Неизвестный номер, нечёткое совпадение, отмена цветового отказа кузовом, сбой сверки кузова.

**Out of scope**

- Смена модели OCR, `frame_interval`, пороги 0.75 / 0.45, вход оператора (G-005), срок хранения снимков.

## Context

**Discovered facts**

- `decide()` начинает с `allow` и только повышает строгость, кроме бывшего присваивания `ALLOW` в `_check_body`.
- Журнал хранит `reasons`, не этапы. Скрытая причина на карточке в базу не попадает.
- Текущий `anpr.yaml` держит `color_check: false` и `fuzzy_decision: review`.

**Product/business constraints**

- Неизвестный номер никогда не даёт `allow`.
- Нечёткое совпадение даёт только `review`.
- Сверка кузова не переписывает цветовой `deny` в `allow`.
- Сбой сверки виден в журнале и не становится тихим `allow`.
- Источник: запрос на этот срез, не вывод из имени файла.

**Assumptions**

- «Журнал» — поле `events.reasons`, не только подпись на карточке. Не подтверждено отдельным регламентом.
- Ночной режим без цвета остаётся: несовпадение цвета при `color_check: false` само по себе не отказ.

## Risk Review

| Risk area | Applies? | Required handling |
| --- | --- | --- |
| Security/privacy | Yes | Сузить `allow`. Ручной пропуск без входа остаётся G-005. |
| Persistent data/migration | No | Схема не меняется. |
| External effects/cost | No | Сеть и модели не вызываются тестами. |
| Compatibility/API | Yes | `fuzzy_decision: deny` больше не даёт `deny`. |
| UX/accessibility | Yes | Карточка Qt, автоматизация — отдельный план, без прогона. |

## Acceptance Criteria

### AC-001: Неизвестный номер не пропускается
- **Scenario:** в базе есть `А123ВС77`, на кадре `Х001ХХ77` либо номер пустой.
- **Action:** `decide(...)`.
- **Expected:** `decision == deny`.
- **Must not:** `allow` или запись чужого id как пропуска.
- **Verification:** `test_decide_matrix_without_body` (`unknown-deny`, `unread-deny`).
- **Priority:** Required

### AC-002: Нечёткость только проверка
- **Scenario:** прочитано `А128ВС77`, в базе `А123ВС77`, `fuzzy_max_distance` 1, конфиг `allow` или `deny`.
- **Action:** `decide(...)`.
- **Expected:** `review`, в `reasons` есть «нечёткое».
- **Must not:** `allow`.
- **Verification:** `test_fuzzy_config_allow_stays_review`, `test_fuzzy_config_deny_stays_review`.
- **Priority:** Required

### AC-003: Кузов не снимает цветовой отказ
- **Scenario:** номер точный, цвет «красный» при базе «белый», близость кузова 0.91, `color_check` включён.
- **Action:** `decide(..., body)`.
- **Expected:** `deny`, причина цвета на месте.
- **Must not:** `allow` и удаление причин с префиксом «цвет ».
- **Verification:** `test_body_match_does_not_override_color_deny`.
- **Priority:** Required

### AC-004: Сбой сверки виден и не пропускает
- **Scenario:** точный номер и совпавший цвет, `BodyMatch` с причиной «сверка кузова не выполнена»; отдельно сеть не создана или вектор чужой длины.
- **Action:** `decide` и `Pipeline._body_match`.
- **Expected:** `review`, та же фраза в `reasons`; причина маски не содержит «эталона» и «кадра».
- **Must not:** `allow` и этап `skip`.
- **Verification:** `test_body_failure_is_journaled_and_not_allow`, `test_missing_embedder_is_not_labeled_as_missing_reference`, `test_shape_mismatch_is_a_failed_check_not_a_missing_frame`.
- **Priority:** Required

## Blocking Decisions

Нет. Ночной `color_check: false` не переключался.

## Verification Plan

| Criterion | Verification evidence | Status |
| --- | --- | --- |
| AC-001 | pytest ids unknown-deny, unread-deny | PASS, 29 тестов |
| AC-002 | два теста fuzzy | PASS |
| AC-003 | test_body_match_does_not_override_color_deny | PASS |
| AC-004 | три теста сбоя | PASS |
