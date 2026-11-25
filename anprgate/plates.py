"""Нормализация и проверка российских автомобильных номеров."""

from __future__ import annotations

import re
from typing import Optional

RU_LETTERS = "АВЕКМНОРСТУХ"

# В номерах РФ используются только кириллические буквы, имеющие латинские двойники.
_LAT_TO_CYR = str.maketrans("ABEKMHOPCTYX", RU_LETTERS)

# Типичные путаницы OCR в зависимости от того, буква или цифра ожидается в позиции.
_DIGIT_AS_LETTER = {"0": "О", "8": "В", "4": "А", "7": "Т"}
_LETTER_AS_DIGIT = {"О": "0", "В": "8", "Т": "7", "А": "4", "D": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "Z": "2", "G": "6"}

# Легковой (в т. ч. двухстрочный): А123ВС77 / А123ВС777.
RU_PRIVATE = re.compile(rf"^[{RU_LETTERS}]\d{{3}}[{RU_LETTERS}]{{2}}\d{{2,3}}$")
# Жёлтый номер такси / общественного транспорта: АА12377 / АА123777.
RU_TAXI = re.compile(rf"^[{RU_LETTERS}]{{2}}\d{{3}}\d{{2,3}}$")
# Шаблоны позиций: L — буква, D — цифра. При равенстве правок выигрывает первый.
_LAYOUTS = ("LDDDLLDD", "LLDDDDD")


def normalize(text: str) -> str:
    """Приводит строку OCR к виду номера РФ: верхний регистр, кириллица, без разделителей."""
    cleaned = re.sub(r"[^0-9A-Za-zА-Яа-яЁё]", "", text or "").upper().replace("Ё", "Е")
    return cleaned.translate(_LAT_TO_CYR)


def _apply_layout(plate: str, layout: str) -> tuple[str, int]:
    chars, changes = [], 0
    for ch, kind in zip(plate, layout):
        fixed = ch
        if kind == "L" and ch.isdigit():
            fixed = _DIGIT_AS_LETTER.get(ch, ch)
        elif kind == "D" and not ch.isdigit():
            fixed = _LETTER_AS_DIGIT.get(ch, ch)
        if (fixed.isdigit()) != (kind == "D"):
            changes += 10  # исправить невозможно — шаблон не подходит
        elif fixed != ch:
            changes += 1
        chars.append(fixed)
    return "".join(chars), changes


def fix_by_layout(plate: str) -> str:
    """Исправляет путаницу букв и цифр по наиболее подходящему шаблону номера РФ."""
    best, best_changes = plate, None
    for base in _LAYOUTS:
        for region_len in (2, 3):
            layout = base + "D" * (region_len - 2)
            if len(layout) != len(plate):
                continue
            fixed, changes = _apply_layout(plate, layout)
            if best_changes is None or changes < best_changes:
                best, best_changes = fixed, changes
    return best if best_changes is not None and best_changes < 10 else plate


def canonical(text: str) -> str:
    return fix_by_layout(normalize(text))


def is_valid_ru(plate: str) -> bool:
    return bool(RU_PRIVATE.match(plate) or RU_TAXI.match(plate))


def levenshtein(a: str, b: str, limit: Optional[int] = None) -> int:
    """Расстояние Левенштейна; при limit досрочно возвращает limit + 1."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if limit is not None and len(a) - len(b) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        if limit is not None and min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]
