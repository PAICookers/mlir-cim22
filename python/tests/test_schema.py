"""Output ABI rejection checks that do not need native compiler libraries."""

import marshal
from collections import OrderedDict

import pytest
from torch.utils import _pytree

from cim22._schema import decode_outputs, encode_outputs


@pytest.mark.parametrize(
    "output",
    [{1: 0}, OrderedDict(value=0)],
    ids=["non-string-key", "custom-container"],
)
def test_unsupported_output_container(output: object) -> None:
    with pytest.raises(ValueError, match="string-keyed dictionaries"):
        encode_outputs(_pytree.tree_structure(output))


@pytest.mark.parametrize(
    "data",
    [
        b"",
        marshal.dumps((99, (0,)), 4),
        marshal.dumps((1, (99, (), ())), 4),
        marshal.dumps((1, (3, ("x", "x"), ((0,), (0,)))), 4),
        marshal.dumps((1, (3, ("x",), ())), 4),
    ],
    ids=["truncated", "version", "container", "duplicate-keys", "key-count"],
)
def test_invalid_output_schema(data: bytes) -> None:
    with pytest.raises(ValueError, match="Invalid CIM22 binary output schema"):
        decode_outputs(data)
