"""Candidate discovery for the DVS CLI and Web application."""

import pytest
import torch
from torch import nn

from cim22.layers import discover_cim_layers, validate_cim_layers


def test_discovery_preserves_names_and_records_shapes() -> None:
    model = nn.Sequential(nn.Conv2d(2, 3, 1), nn.ReLU()).eval()
    assert discover_cim_layers(model) == [
        {
            "name": "0",
            "type": "Conv2d",
            "weight_shape": [3, 2, 1, 1],
            "input_shape": [],
            "output_shape": [],
        }
    ]
    candidates = discover_cim_layers(model, (torch.ones(1, 2, 4, 4),))
    assert candidates[0]["input_shape"] == [1, 2, 4, 4]
    assert candidates[0]["output_shape"] == [1, 3, 4, 4]
    assert not model[0]._forward_hooks


def test_discovery_removes_hooks_on_model_error() -> None:
    model = nn.Sequential(nn.Linear(2, 2)).eval()
    with pytest.raises(RuntimeError):
        discover_cim_layers(model, (torch.ones(1, 3),))
    assert not model[0]._forward_hooks


@pytest.mark.parametrize(
    ("selected", "message"),
    [
        ([], "at least one"),
        (["0", "0"], "unique"),
        (["missing"], "Unknown"),
        (["1"], "Unsupported"),
        (["2"], "leaf"),
        ("0", "non-empty strings"),
        ([None], "non-empty strings"),
        ([""], "non-empty strings"),
    ],
)
def test_candidate_selection_rejects_invalid_names(
    selected: object, message: str
) -> None:
    model = nn.Sequential(
        nn.Linear(2, 2), nn.ReLU(), nn.Sequential(nn.Linear(2, 2))
    )
    with pytest.raises(ValueError, match=message):
        validate_cim_layers(model, selected)


def test_candidate_selection_and_shared_module_rejection() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    assert validate_cim_layers(model, None) == ()
    assert validate_cim_layers(model, ["0", "1"]) == ("0", "1")
    model.alias = model[0]
    with pytest.raises(ValueError, match="same module"):
        validate_cim_layers(model, ["alias"])
