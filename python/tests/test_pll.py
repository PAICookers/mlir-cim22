"""Fixed supplier table, labels, and frequency-selection boundaries."""

import warnings
from collections import Counter
from dataclasses import FrozenInstanceError

import pytest

from cim22.pll import PLL_CLOCK_TABLE, PLLClockSetting, select_pll_clock


@pytest.mark.parametrize(
    ("target", "actual", "serial", "status"),
    [
        (15.75, 15.75, "507D44", "available"),
        (15.9, 15.9, "D13F44", "available"),
        (20, 20, "88B624", "available"),
        (24, 24, "2828A4", "available"),
        (128, 128, "3C08F4", "available"),
        (260, 259.2, "D41354", "maybe available"),
        (284, 283.2, "E813A4", "maybe available"),
        (484, 480, "4C0134", "maybe available"),
        (1000, 1008, "A40294", "maybe available"),
    ],
)
def test_supplier_presets_and_warning_policy(
    target: float, actual: float, serial: str, status: str
) -> None:
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        setting = select_pll_clock(target)
    assert setting.target_frequency_mhz == target
    assert setting.actual_frequency_mhz == actual
    assert setting.serial_hex == serial
    assert setting.status == status
    if status == "maybe available":
        assert len(emitted) == 1
        assert emitted[0].category is RuntimeWarning
        message = str(emitted[0].message)
        assert "maybe available" in message
        assert "may be unstable" in message
        assert f"{target:g} MHz" in message
        assert f"{actual:g} MHz" in message
        assert serial in message
    else:
        assert not emitted


def test_lookup_preserves_every_row_and_warns_only_for_maybe_available() -> (
    None
):
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        for row in PLL_CLOCK_TABLE:
            assert select_pll_clock(row.target_frequency_mhz) is row
    assert len(emitted) == 70
    assert all(item.category is RuntimeWarning for item in emitted)


def test_table_retains_all_rows_and_labels() -> None:
    assert len(PLL_CLOCK_TABLE) == 199
    assert len({row.target_frequency_mhz for row in PLL_CLOCK_TABLE}) == 199
    assert Counter(row.status for row in PLL_CLOCK_TABLE) == {
        "available": 129,
        "maybe available": 70,
    }
    assert len({row.serial_hex for row in PLL_CLOCK_TABLE}) == 148
    with pytest.raises(FrozenInstanceError):
        PLL_CLOCK_TABLE[0].serial_hex = "000000"


@pytest.mark.parametrize("row", PLL_CLOCK_TABLE)
def test_stored_serial_fields_agree_with_table_frequencies(
    row: PLLClockSetting,
) -> None:
    # Independent decoder verifies literal table entries; selection never uses it.
    word = int(row.serial_hex, 16)
    nf = ((word >> 18) & 63) + 1
    nr = ((word >> 14) & 15) + 1
    od = ((word >> 10) & 15) + 1
    assert word & 15 == 4
    assert (word >> 4) & 63 == nf - 1
    assert 24 * nf / nr >= 250
    assert row.actual_frequency_mhz == pytest.approx(
        24 * nf / nr / od, abs=1e-9, rel=0
    )


@pytest.mark.parametrize(
    "value",
    [0, -1, 15.74, 26, 283.2, 1001, float("inf"), float("nan"), 10**400],
)
def test_unlisted_frequencies_are_not_synthesized(value: float) -> None:
    # 26 MHz is physically synthesizable but absent from the supplier target rows.
    with pytest.raises(ValueError, match="not a listed target"):
        select_pll_clock(value)


@pytest.mark.parametrize("value", [True, False, "24", None, object()])
def test_frequency_rejects_non_numeric_inputs(value: object) -> None:
    with pytest.raises(TypeError, match="clock_frequency_mhz"):
        select_pll_clock(value)
