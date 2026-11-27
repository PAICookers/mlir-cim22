"""JAX callable and artifact contracts through the released ONNX frontend."""

from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from cim22 import Session
from cim22.runtime import MODEL_SUFFIX


@pytest.fixture(scope="module")
def frontend() -> ModuleType:
    pytest.importorskip("jax2onnx")
    from cim22 import jax as frontend

    return frontend


@pytest.mark.integration
@pytest.mark.usefixtures("native_toolchain")
def test_static_signature_and_tree(
    frontend: ModuleType, tmp_path: Path
) -> None:
    import jax
    import jax.numpy as jnp

    def model(x, y):
        return {"sum": x + y, "other": (x - y, x.astype(jnp.int8))}

    spec = jax.ShapeDtypeStruct((2,), jnp.int32)
    x, y = jnp.array([5, -2], jnp.int32), jnp.array([7, -1], jnp.int32)
    with frontend.compile(
        model,
        (spec, spec),
        options={"runtime": "simulator", "artifact_dir": tmp_path},
    ) as compiled:
        out = compiled(x, y)
        assert jax.tree.structure(out) == jax.tree.structure(model(x, y))
        assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(out))
        for actual, expected in zip(
            jax.tree.leaves(out), jax.tree.leaves(model(x, y)), strict=True
        ):
            np.testing.assert_array_equal(actual, expected, strict=True)
        with pytest.raises(TypeError):
            compiled(x.astype(jnp.int8), y)
    (library,) = tmp_path.glob(f"graph-*{MODEL_SUFFIX}")
    with Session(library) as session:
        results = session(np.asarray(x), np.asarray(y))
        for actual, expected in zip(results, jax.tree.leaves(out), strict=True):
            np.testing.assert_array_equal(actual, expected, strict=True)


@pytest.mark.integration
@pytest.mark.usefixtures("native_toolchain")
def test_int8_contraction_freezes_weights(frontend: ModuleType) -> None:
    import jax
    import jax.numpy as jnp

    weights = jnp.ones((64, 16), dtype=jnp.int8)
    x = jnp.full((2, 64), 3, dtype=jnp.int8)

    def model(x):
        return jnp.matmul(x, weights, preferred_element_type=jnp.int32)

    expected = np.asarray(model(x))
    with frontend.compile(
        model,
        (jax.ShapeDtypeStruct(x.shape, x.dtype),),
        options={"runtime": "simulator"},
    ) as compiled:
        weights = jnp.zeros_like(weights)
        np.testing.assert_array_equal(compiled(x), expected, strict=True)
        assert compiled.session.dispatch_count > 0


def test_signature_validation(frontend: ModuleType) -> None:
    import jax
    import jax.numpy as jnp

    for spec in (
        jax.ShapeDtypeStruct((2,), jnp.float16),
        jax.ShapeDtypeStruct((0,), jnp.float32),
    ):
        with pytest.raises(ValueError, match="JAX inputs"):
            frontend.compile(lambda x: x, (spec,))
    with pytest.raises(ValueError, match="JAX outputs"):
        frontend.compile(
            lambda x: x > 0, (jax.ShapeDtypeStruct((2,), jnp.int8),)
        )
