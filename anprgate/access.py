"""Правила допуска: номер в базе → совпадение цвета → решение."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .config import AccessConfig
from .db import Vehicle
from .models.base import ColorEstimate
from .models.color import colors_compatible, delta_e
from .plates import levenshtein

ALLOW, DENY, REVIEW = "allow", "deny", "review"
_SEVERITY = {ALLOW: 0, REVIEW: 1, DENY: 2}

DECISION_TITLES = {ALLOW: "ПРОПУСТИТЬ", DENY: "ОТКАЗ", REVIEW: "ПРОВЕРКА ОПЕРАТОРОМ"}

STAGE_OK, STAGE_WARN, STAGE_FAIL, STAGE_SKIP = "ok", "warn", "fail", "skip"
STAGE_MARKS = {STAGE_OK: "✓", STAGE_WARN: "?", STAGE_FAIL: "✗", STAGE_SKIP: "—"}


@dataclass
class StageResult:
    title: str
    status: str
    detail: str

    def line(self) -> str:
        return f"{STAGE_MARKS[self.status]} {self.title}: {self.detail}"


@dataclass
class BodyMatch:
    """Близость кузова к эталону той же записи в базе. score — косинус, 0..1."""

    score: Optional[float]
    frames: int = 0
    references: int = 0
    reason: Optional[str] = None


@dataclass
class AccessDecision:
    decision: str
    reasons: List[str] = field(default_factory=list)
    vehicle: Optional[Vehicle] = None
    match_distance: Optional[int] = None
    color_delta_e: Optional[float] = None
    body_score: Optional[float] = None
    stages: List[StageResult] = field(default_factory=list)

    def escalate(self, decision: str, reason: str) -> None:
        if _SEVERITY[decision] > _SEVERITY[self.decision]:
            self.decision = decision
        self.reasons.append(reason)


def find_vehicle(plate: str, vehicles: Sequence[Vehicle], max_distance: int):
    """Точное совпадение, иначе ближайший номер в пределах max_distance (единственный)."""
    best, best_dist, ambiguous = None, max_distance + 1, False
    for vehicle in vehicles:
        dist = levenshtein(plate, vehicle.plate, limit=max_distance)
        if dist == 0:
            return vehicle, 0
        if dist < best_dist:
            best, best_dist, ambiguous = vehicle, dist, False
        elif dist == best_dist:
            ambiguous = True
    if best is None or ambiguous:
        return None, None
    return best, best_dist


def decide(
    plate: Optional[str],
    plate_valid: bool,
    color: ColorEstimate,
    vehicles: Sequence[Vehicle],
    cfg: AccessConfig,
    body: Optional[BodyMatch] = None,
) -> AccessDecision:
    """Номер, затем цвет найденного ТС, затем близость кузова к эталону этой записи."""
    result = AccessDecision(ALLOW)
    plate_stage = _check_plate(result, plate, plate_valid, vehicles, cfg)
    result.stages.append(plate_stage)

    if plate_stage.status == STAGE_FAIL:
        result.stages.append(StageResult("Цвет", STAGE_SKIP, "не проверялся — номер не прошёл"))
        if cfg.body_check:
            result.stages.append(StageResult("Кузов", STAGE_SKIP, "не проверялся — номер не прошёл"))
    else:
        if not cfg.color_check:
            result.stages.append(StageResult("Цвет", STAGE_SKIP, "проверка цвета отключена"))
        else:
            result.stages.append(_check_color(result, color, result.vehicle, cfg))
        if cfg.body_check:
            result.stages.append(_check_body(result, body, cfg))

    if result.decision == ALLOW:
        result.reasons.append(_allow_reason(result, cfg))
    return result


def _allow_reason(result: AccessDecision, cfg: AccessConfig) -> str:
    body = next((s for s in result.stages if s.title == "Кузов"), None)
    if body and body.status == STAGE_OK:
        return "номер и кузов совпадают с эталоном"
    if body and body.status == STAGE_WARN:
        if cfg.color_check:
            return "номер и цвет совпадают, кузов эталоном не подтверждён"
        return "номер найден, кузов не подтверждён"
    if cfg.color_check:
        return "номер и цвет совпадают с базой"
    return "номер найден в базе"


def _check_body(result: AccessDecision, body: Optional[BodyMatch], cfg: AccessConfig) -> StageResult:
    if body is None or body.score is None:
        reason = body.reason if body and body.reason else "нет эталона кузова"
        failed = bool(body and body.reason and "не выполнена" in body.reason)
        if failed or not cfg.color_check:
            result.escalate(REVIEW, reason)
            return StageResult("Кузов", STAGE_WARN, reason)
        return StageResult("Кузов", STAGE_SKIP, reason)

    result.body_score = body.score
    text = f"близость {body.score:.2f} к эталону ({body.frames} кадр.)"
    if body.score >= cfg.body_match:
        note = ", кузов совпадает"
        if result.decision == DENY:
            note += ", отказ по цвету сохранён"
        return StageResult("Кузов", STAGE_OK, text + note)
    if body.score < cfg.body_alarm:
        result.escalate(DENY, f"кузов не совпадает с эталоном (близость {body.score:.2f})")
        return StageResult("Кузов", STAGE_FAIL, text + ", это другая машина")

    color_stage = next(s for s in result.stages if s.title == "Цвет")
    if color_stage.status == STAGE_SKIP:
        result.escalate(REVIEW, f"кузов не подтверждён (близость {body.score:.2f})")
        return StageResult("Кузов", STAGE_WARN, text + ", нужна проверка")
    if color_stage.status == STAGE_FAIL:
        result.escalate(DENY, f"кузов не подтверждён (близость {body.score:.2f}) и цвет не совпал")
        return StageResult("Кузов", STAGE_FAIL, text + ", не подтверждён")
    if color_stage.status == STAGE_WARN:
        result.escalate(REVIEW, f"кузов не подтверждён (близость {body.score:.2f}), цвет не определён")
        return StageResult("Кузов", STAGE_WARN, text + ", нужна проверка")
    return StageResult("Кузов", STAGE_WARN, text + ", цвет совпал, кузов не подтверждён")


def _check_plate(result: AccessDecision, plate: Optional[str], plate_valid: bool,
                 vehicles: Sequence[Vehicle], cfg: AccessConfig) -> StageResult:
    if not plate:
        result.escalate(DENY, "номер не распознан")
        return StageResult("Номер", STAGE_FAIL, "не распознан")

    vehicle, distance = find_vehicle(plate, vehicles, cfg.fuzzy_max_distance)
    result.vehicle, result.match_distance = vehicle, distance
    if vehicle is None:
        result.escalate(DENY, "номер отсутствует в базе")
        return StageResult("Номер", STAGE_FAIL, f"{plate} отсутствует в базе")
    if not vehicle.active:
        result.escalate(DENY, "пропуск ТС деактивирован")
        return StageResult("Номер", STAGE_FAIL, f"{vehicle.plate} в базе, но пропуск отключён")

    status, notes = STAGE_OK, []
    if cfg.require_valid_format and not plate_valid:
        result.escalate(REVIEW, "номер не соответствует формату РФ")
        status = STAGE_WARN
        notes.append("формат не РФ")
    if distance:
        # Нечёткое совпадение всегда уходит оператору: allow открыл бы чужой пропуск, deny прячет сомнение.
        result.escalate(REVIEW, f"нечёткое совпадение с {vehicle.plate} (отличий: {distance})")
        status = STAGE_WARN
        notes.append(f"прочитано {plate}, отличий: {distance}")
    detail = f"{vehicle.plate} найден в базе" + (f" ({'; '.join(notes)})" if notes else "")
    return StageResult("Номер", status, detail)


def _check_color(result: AccessDecision, color: ColorEstimate, vehicle: Vehicle,
                 cfg: AccessConfig) -> StageResult:
    if color.name is None:
        result.escalate(cfg.color_unknown_decision, f"цвет не определён: {color.reason}")
        return StageResult("Цвет", STAGE_WARN, f"не определён ({color.reason}), в базе «{vehicle.color}»")

    by_name = colors_compatible(color.name, vehicle.color)
    by_lab = False
    if vehicle.color_lab and color.lab:
        result.color_delta_e = delta_e(color.lab, vehicle.color_lab)
        by_lab = result.color_delta_e <= cfg.color_delta_e_max
    delta = f", ΔE {result.color_delta_e:.0f}" if result.color_delta_e is not None else ""
    if not (by_name or by_lab):
        result.escalate(DENY, f"цвет «{color.name}» не совпадает с «{vehicle.color}» из базы")
        return StageResult("Цвет", STAGE_FAIL, f"на камере «{color.name}», в базе «{vehicle.color}»{delta}")
    return StageResult("Цвет", STAGE_OK, f"на камере «{color.name}», в базе «{vehicle.color}»{delta}")
