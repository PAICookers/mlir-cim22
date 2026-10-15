"""Protobuf output trees embedded in a model library, using built-in containers."""

from google.protobuf.message import DecodeError
from torch.utils import _pytree

from cim22._generated import cim22_model_pb2 as proto


def _encode(
    spec: _pytree.TreeSpec, node: proto.OutputNode, ordinal: int
) -> int:
    if spec.is_leaf():
        node.tensor_leaf = ordinal
        return ordinal + 1
    if spec.type not in (tuple, list, dict) or (
        spec.type is dict
        and not all(isinstance(key, str) for key in spec.context)
    ):
        raise ValueError(
            "Outputs require tensors, tuples, lists, or string-keyed dictionaries"
        )
    if spec.type is dict:
        node.dictionary.SetInParent()
        if len(set(spec.context)) != len(spec.context):
            raise ValueError("Duplicate output dictionary keys")
        for key, child in zip(spec.context, spec.children(), strict=True):
            entry = node.dictionary.entries.add(key=key)
            ordinal = _encode(child, entry.value, ordinal)
    else:
        sequence = node.tuple if spec.type is tuple else node.list
        sequence.SetInParent()
        for child in spec.children():
            ordinal = _encode(child, sequence.children.add(), ordinal)
    return ordinal


def _decode(
    node: proto.OutputNode, ordinal: int
) -> tuple[_pytree.TreeSpec, int]:
    kind = node.WhichOneof("node")
    if kind == "tensor_leaf":
        if node.tensor_leaf != ordinal:
            raise ValueError("Invalid output tensor leaf ordinal")
        return _pytree.tree_structure(0), ordinal + 1
    if kind not in ("tuple", "list", "dictionary"):
        raise ValueError("Missing or unsupported output schema node")
    keys = None
    if kind == "dictionary":
        entries = node.dictionary.entries
        keys = [entry.key for entry in entries]
        if len(set(keys)) != len(keys) or any(
            not entry.HasField("key") or not entry.HasField("value")
            for entry in entries
        ):
            raise ValueError("Invalid output dictionary entries")
        nodes = [entry.value for entry in entries]
    else:
        nodes = getattr(node, kind).children
    children = []
    for child in nodes:
        spec, ordinal = _decode(child, ordinal)
        children.append(spec)
    return (
        _pytree.TreeSpec(
            {"tuple": tuple, "list": list, "dictionary": dict}[kind],
            keys,
            children,
        ),
        ordinal,
    )


def encode_outputs(spec: _pytree.TreeSpec) -> bytes:
    """Encode built-in containers with tensor ordinals in flatten order."""
    schema = proto.OutputSchema(version=1)
    _encode(spec, schema.root, 0)
    data = schema.SerializeToString()
    # Match the enclosing native model's parser depth before compilation.
    try:
        proto.ModelArtifact.FromString(
            proto.ModelArtifact(output_schema=schema).SerializeToString()
        )
    except DecodeError as error:
        raise ValueError("Output schema exceeds model parser limit") from error
    return data


def decode_outputs(data: bytes) -> _pytree.TreeSpec:
    """Restore and validate the embedded output tree; old marshal is rejected."""
    try:
        schema = proto.OutputSchema.FromString(data)
        if schema.version != 1 or not schema.HasField("root"):
            raise ValueError(
                "Unsupported or missing output schema version/root"
            )
        return _decode(schema.root, 0)[0]
    except (DecodeError, ValueError) as error:
        raise ValueError(
            f"Invalid CIM22 Protobuf output schema; recompile the model: {error}"
        ) from error
