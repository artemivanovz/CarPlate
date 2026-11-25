"""Кто в базе попадает на «вход разрешён»: белый список и последние allow-события.

Запуск: .venv\\Scripts\\python.exe scripts\\db_show.py [последние N событий, по умолчанию 8]
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anprgate.config import load_config  # noqa: E402


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    cfg = load_config()
    con = sqlite3.connect(cfg.db_path)
    con.row_factory = sqlite3.Row

    print(f"БД: {cfg.db_path}")
    print("\n== Белый список (active=1 → вход разрешён) ==")
    rows = list(con.execute(
        "SELECT plate, color, owner, note, created_at FROM vehicles WHERE active = 1 ORDER BY plate"
    ))
    if not rows:
        print("(пусто)")
    for row in rows:
        extra = " ".join(f"{v}" for v in (row["owner"], row["note"]) if v)
        print(f"{row['plate']:<10} цвет={row['color']:<12} {extra}".rstrip())

    print("\n== Последние события «вход разрешён» (allow) ==")
    events = list(con.execute(
        "SELECT id, ts, plate, color_detected, vehicle_id, match_distance, reasons "
        "FROM events WHERE decision = 'allow' ORDER BY id DESC LIMIT ?",
        (limit,),
    ))
    if not events:
        print("(пусто)")
    for ev in events:
        print(f"#{ev['id']} {ev['ts']}  номер={ev['plate'] or '—'}  цвет={ev['color_detected'] or '—'}  "
              f"vehicle_id={ev['vehicle_id']}  совпадение={ev['match_distance']}")
    allow_count = con.execute("SELECT COUNT(*) FROM events WHERE decision='allow'").fetchone()[0]
    print(f"\nИтого в журнале allow-записей: {allow_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())