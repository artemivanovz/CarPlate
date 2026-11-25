import pytest

from anprgate.db import Database
from scripts.body_separation_check import main, reference_vehicles


def test_demo_write_requires_explicit_targets_before_loading_models():
    with pytest.raises(SystemExit) as error:
        main(["--write"])
    assert error.value.code == 2


def test_reference_targets_are_selected_by_plate_instead_of_color_order(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    try:
        unrelated = db.add_vehicle("А000АА77", "красный")
        gray_id = db.add_vehicle("А001АА799", "серебристый")
        red_id = db.add_vehicle("Н991КК797", "красный")
        gray, red = reference_vehicles(db, "A001AA799", "H991KK797")
        assert (gray.id, red.id) == (gray_id, red_id)
        assert unrelated not in (gray.id, red.id)
        with pytest.raises(ValueError):
            reference_vehicles(db, gray.plate, gray.plate)
        db.update_vehicle(red.id, active=False)
        with pytest.raises(ValueError):
            reference_vehicles(db, gray.plate, red.plate)
    finally:
        db.close()
