"""Native loading rejects malformed Protobuf before invoking model code."""

from typing import TYPE_CHECKING

import pytest

from cim22.runtime import Session

from ._generated import cim22_model_pb2 as proto

if TYPE_CHECKING:
    from .conftest import ModelLibrary


pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("native_runtime"),
]


@pytest.mark.parametrize("with_transaction", [False, True], ids=["host", "cim"])
def test_protobuf_interoperability(
    model_library: "ModelLibrary",
    with_transaction: bool,
    host_artifact: proto.ModelArtifact,
    transaction_artifact: proto.ModelArtifact,
) -> None:
    model = transaction_artifact if with_transaction else host_artifact
    shared = model_library(model.SerializeToString() + b"\xa0\x06\x01", 1)
    with Session(shared) as session:
        assert session.num_transactions == int(with_transaction)
        assert session.input_shapes == ()
        assert session.output_shapes == ((),)


@pytest.mark.parametrize(
    "failure",
    [
        "empty",
        "truncated",
        "dtype",
        "shape",
        "overflow",
        "missing-transaction",
        "duplicate-kernel",
    ],
)
def test_invalid_model_metadata(
    model_library: "ModelLibrary",
    failure: str,
    host_artifact: proto.ModelArtifact,
    transaction_artifact: proto.ModelArtifact,
) -> None:
    model = host_artifact
    match failure:
        case "dtype":
            model.outputs[0].data_type = 99
        case "shape":
            model.outputs[0].dimensions.append(-1)
        case "overflow":
            model.outputs[0].dimensions.extend([2**62, 4])
        case "missing-transaction":
            model.kernels.add(kernel_name="missing")
        case "duplicate-kernel":
            model = transaction_artifact
            model.kernels.add().CopyFrom(model.kernels[0])
    data = model.SerializeToString()
    if failure == "empty":
        data = b""
    elif failure == "truncated":
        data = data[:-1]
    with pytest.raises(RuntimeError, match="Protobuf model.*recompile"):
        Session(model_library(data, 1))


@pytest.mark.parametrize(
    "failure",
    [
        "profile",
        "missing-id",
        "duplicate-group",
        "duplicate-work",
        "core-slot",
        "route-length",
        "weight-words",
        "weight-values",
        "weight-exponents",
        "weight-dtype",
        "weight-work-route",
        "missing-input",
        "input-work",
        "input-slot",
        "duplicate-input",
        "readback-address",
        "readback-work",
        "readback-work-route",
        "readback-test-core",
        "datatype-mismatch",
        "packet-kind",
        "packet-dtype",
        "packet-address",
        "packet-work",
        "packet-route",
        "missing-packet-id",
        "missing-exponent-packet",
        "packet-route-binding",
        "packet-test-core",
        "readback-sequence",
        "flits",
    ],
)
def test_invalid_transaction(
    model_library: "ModelLibrary",
    failure: str,
    transaction_artifact: proto.ModelArtifact,
) -> None:
    model = transaction_artifact
    transaction = model.kernels[0].transaction
    group = transaction.groups[0]
    weight = transaction.weights[0]
    binding = transaction.inputs[0]
    readback = transaction.readbacks[0]
    packet = transaction.packets[0]
    match failure:
        case "profile":
            transaction.target_profile = "unknown"
        case "missing-id":
            group.works[0].ClearField("work_id")
        case "duplicate-group":
            transaction.groups.add().CopyFrom(group)
        case "duplicate-work":
            group.works.add().CopyFrom(group.works[0])
        case "core-slot":
            group.works[0].core_slot = 20
        case "route-length":
            group.works[0].route.pop()
        case "weight-words":
            weight.words.pop()
        case "weight-values":
            weight.values = weight.values[:-1]
        case "weight-exponents":
            weight.exponents = weight.exponents[:-1]
        case "weight-dtype":
            weight.data_type = proto.UNSPECIFIED
        case "weight-work-route":
            group.works[0].route[0] = 1
        case "missing-input":
            transaction.ClearField("inputs")
        case "input-work":
            binding.work_id = 9
        case "input-slot":
            binding.input_slot = -1
        case "duplicate-input":
            transaction.inputs.add().CopyFrom(binding)
        case "readback-address":
            readback.output_cache_address = 8
        case "readback-work":
            readback.work_id = 9
        case "readback-work-route":
            readback.route[0] = 1
            for read_packet in transaction.packets[3:]:
                read_packet.route[0] = 1
        case "readback-test-core":
            readback.test_core.pop()
        case "datatype-mismatch":
            readback.data_type = proto.INT8
        case "packet-kind":
            packet.kind = 99
        case "packet-dtype":
            packet.data_type = 99
        case "packet-address":
            transaction.packets[-1].cache_address = 6
        case "packet-work":
            packet.work_id = 9
        case "packet-route":
            packet.route.pop()
        case "missing-packet-id":
            packet.ClearField("group_id")
        case "missing-exponent-packet":
            del transaction.packets[1]
        case "packet-route-binding":
            packet.route[0] = 1
        case "packet-test-core":
            transaction.packets[3].test_core[0] = 1
        case "readback-sequence":
            transaction.packets[4].kind = proto.WORK
        case "flits":
            transaction.ClearField("flits")
    with pytest.raises(RuntimeError, match="Protobuf model.*recompile"):
        Session(model_library(model.SerializeToString(), 1))


def test_invalid_native_abi(
    model_library: "ModelLibrary", host_artifact: proto.ModelArtifact
) -> None:
    with pytest.raises(RuntimeError, match="ABI"):
        Session(model_library(host_artifact.SerializeToString(), 99))


@pytest.mark.parametrize(
    "duplicate", [False, True], ids=["distinct", "duplicate"]
)
def test_physical_macro_placement(
    model_library: "ModelLibrary",
    duplicate: bool,
    transaction_artifact: proto.ModelArtifact,
) -> None:
    model = transaction_artifact
    transaction = model.kernels[0].transaction
    works = transaction.groups[0].works
    work = works.add()
    work.CopyFrom(works[0])
    work.work_id = 1
    work.macro_slot = 0 if duplicate else 1
    for bindings in (
        transaction.weights,
        transaction.inputs,
        transaction.readbacks,
    ):
        binding = bindings.add()
        binding.CopyFrom(bindings[0])
        binding.work_id = 1
        binding.macro_slot = work.macro_slot
    transaction.inputs[1].input_slot = 1
    transaction.inputs[1].input_cache_address = 1
    transaction.readbacks[1].output_cache_address = 6
    for original in list(transaction.packets):
        packet = transaction.packets.add()
        packet.CopyFrom(original)
        packet.work_id = 1
        packet.macro_slot = work.macro_slot
        if packet.kind == proto.OUTPUT_CACHE_READ:
            packet.cache_address = 6
    shared = model_library(model.SerializeToString(), 1)
    if duplicate:
        with pytest.raises(RuntimeError, match="duplicate work, placement"):
            Session(shared)
    else:
        with Session(shared) as session:
            assert session.num_transactions == 1


@pytest.mark.parametrize(
    ("version", "backend", "cache", "valid"),
    [
        (None, None, None, True),
        (99, proto.USB, 7, True),
        (99, proto.USB, None, False),
        (1, 99, 0, False),
        (1, proto.USB, 8, False),
    ],
)
def test_development_metadata_contract(
    model_library: "ModelLibrary",
    transaction_artifact: proto.ModelArtifact,
    version: int | None,
    backend: int | None,
    cache: int | None,
    valid: bool,
) -> None:
    model = transaction_artifact
    if version is None:
        model.ClearField("schema_version")
    else:
        model.schema_version = version
    if backend is None:
        model.ClearField("backend")
    else:
        model.backend = backend
    binding = model.kernels[0].transaction.inputs[0]
    if cache is None:
        binding.ClearField("input_cache_address")
    else:
        binding.input_cache_address = cache
    shared = model_library(model.SerializeToString(), 1)
    if valid:
        with Session(shared) as session:
            assert session.num_transactions == 1
    else:
        with pytest.raises(RuntimeError, match="Protobuf model.*recompile"):
            Session(shared)
