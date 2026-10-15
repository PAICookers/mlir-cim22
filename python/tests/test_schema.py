"""Output ABI interoperability and rejection without native compiler libraries."""

import marshal
from collections import OrderedDict

import pytest
from torch.utils import _pytree

from cim22._generated import cim22_model_pb2 as proto
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
    "output",
    [0, (), [], {}, {"z": (0, [0]), "a": 0, "": []}],
    ids=["leaf", "empty-tuple", "empty-list", "empty-dict", "nested-ordered"],
)
def test_output_schema_round_trip(output: object) -> None:
    spec = _pytree.tree_structure(output)
    encoded = encode_outputs(spec)
    assert decode_outputs(encoded) == spec
    schema = proto.OutputSchema.FromString(encoded)
    assert schema.version == 1
    if isinstance(output, dict) and output:
        assert [entry.key for entry in schema.root.dictionary.entries] == [
            "z",
            "a",
            "",
        ]
        assert schema.root.dictionary.entries[1].value.tensor_leaf == 2
    # Field 100 is unknown non-critical metadata, safely ignored.
    assert decode_outputs(encoded + b"\xa0\x06\x01") == spec


def invalid_schemas() -> list[bytes]:
    schema = proto.OutputSchema(version=1)
    values = [
        b"",
        b"\x08",
        marshal.dumps((1, (0,)), 4),
        proto.OutputSchema(version=99).SerializeToString(),
        schema.SerializeToString(),
    ]
    schema.root.SetInParent()
    values.append(schema.SerializeToString())
    schema.root.tensor_leaf = 1
    values.append(schema.SerializeToString())
    schema.root.Clear()
    for ordinal in range(2):
        schema.root.dictionary.entries.add(key="x").value.tensor_leaf = ordinal
    values.append(schema.SerializeToString())
    schema.root.dictionary.entries[1].key = "y"
    schema.root.dictionary.entries[1].ClearField("value")
    values.append(schema.SerializeToString())
    schema.root.dictionary.entries[1].value.tensor_leaf = 1
    schema.root.dictionary.entries[1].ClearField("key")
    values.append(schema.SerializeToString())
    return values


@pytest.mark.parametrize(
    "data",
    invalid_schemas(),
    ids=[
        "empty",
        "truncated",
        "marshal",
        "version",
        "missing-root",
        "missing-node",
        "leaf-ordinal",
        "duplicate-keys",
        "missing-value",
        "missing-key",
    ],
)
def test_invalid_output_schema(data: bytes) -> None:
    with pytest.raises(ValueError, match="Protobuf output schema; recompile"):
        decode_outputs(data)


@pytest.mark.parametrize(
    ("lists", "dictionary", "accepted"),
    [(49, False, True), (47, True, True), (48, True, False)],
    ids=["list-limit", "dict-below-limit", "dict-exceeds-model-limit"],
)
def test_enclosing_model_parser_limit(
    lists: int, dictionary: bool, accepted: bool
) -> None:
    output = {"x": 0} if dictionary else 0
    for _ in range(lists):
        output = [output]
    spec = _pytree.tree_structure(output)
    if accepted:
        assert decode_outputs(encode_outputs(spec)) == spec
    else:
        with pytest.raises(ValueError, match="model parser limit"):
            encode_outputs(spec)
