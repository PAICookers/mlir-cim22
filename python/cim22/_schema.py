"""Binary output-container metadata embedded in a Linux model library.

Only built-in containers and tensor leaves are represented. Python 3.12 marshal
encodes primitive tuples/strings without importing classes or using JSON/pickle.
The native model library is trusted executable code, not an untrusted input.
"""

import marshal

from torch.utils import _pytree

type _Node = tuple[int] | tuple[int, tuple[str, ...], tuple[_Node, ...]]


def _encode(spec: _pytree.TreeSpec) -> _Node:
    if spec.is_leaf():
        return (0,)
    kinds = {tuple: 1, list: 2, dict: 3}
    if spec.type not in kinds or (
        spec.type is dict
        and not all(isinstance(key, str) for key in spec.context)
    ):
        raise ValueError(
            "Outputs require tensors, tuples, lists, or string-keyed dictionaries"
        )
    keys = tuple(spec.context) if spec.type is dict else ()
    return kinds[spec.type], keys, tuple(_encode(s) for s in spec.children())


def _decode(data: object) -> _pytree.TreeSpec:
    if data == (0,):
        return _pytree.tree_structure(0)
    if not isinstance(data, tuple) or len(data) != 3:
        raise ValueError("Invalid output schema node")
    kind, keys, children = data
    if type(kind) is not int or kind not in (1, 2, 3):
        raise ValueError("Invalid output schema container")
    if not isinstance(keys, tuple) or not isinstance(children, tuple):
        raise ValueError("Invalid output schema fields")
    if kind == 3:
        if (
            len(keys) != len(children)
            or not all(isinstance(key, str) for key in keys)
            or len(set(keys)) != len(keys)
        ):
            raise ValueError("Invalid output dictionary keys")
    elif keys:
        raise ValueError("Unexpected output schema keys")
    return _pytree.TreeSpec(
        {1: tuple, 2: list, 3: dict}[kind],
        list(keys) if kind == 3 else None,
        [_decode(child) for child in children],
    )


def encode_outputs(spec: _pytree.TreeSpec) -> bytes:
    """Encode the supported output tree, with a separate schema version."""
    return marshal.dumps((1, _encode(spec)), 4)


def decode_outputs(data: bytes) -> _pytree.TreeSpec:
    """Restore only built-in tensor containers from the embedded binary schema."""
    try:
        version, tree = marshal.loads(data)
        if type(version) is not int or version != 1:
            raise ValueError("Unsupported output schema version")
        return _decode(tree)
    except (EOFError, TypeError, ValueError) as error:
        raise ValueError("Invalid CIM22 binary output schema") from error
