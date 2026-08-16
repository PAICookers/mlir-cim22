import re
import sys
import unittest

import route_oracle as oracle

# Frozen expected routes are listed, never generated with the production policy.
ROUTES = (
    (0, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, 0),
    (0, 2, 0, 0, 0, 0),
    (0, 3, 0, 0, 0, 0),
    (0, 4, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0),
    (1, 0, 0, 0, 0, 0),
    (1, 1, 0, 0, 0, 0),
    (1, 2, 0, 0, 0, 0),
    (1, 3, 0, 0, 0, 0),
    (0, 0, 2, 0, 0, 0),
    (1, 0, 1, 0, 0, 0),
    (2, 0, 0, 0, 0, 0),
    (2, 1, 0, 0, 0, 0),
    (2, 2, 0, 0, 0, 0),
    (0, 0, 3, 0, 0, 0),
    (1, 0, 2, 0, 0, 0),
    (2, 0, 1, 0, 0, 0),
    (3, 0, 0, 0, 0, 0),
    (3, 1, 0, 0, 0, 0),
)


def valid_records():
    records = []
    for core_slot, (coordinate, route) in enumerate(zip(oracle.PROFILE_SLOTS, ROUTES)):
        for macro_slot in (0, 1):
            records.append(oracle.Mapping(
                len(records), core_slot, macro_slot, coordinate,
                oracle.LOWER_LEFT, oracle.LOWER_LEFT, coordinate, route))
    return tuple(records)


def render_dump(records=None, omit_attrs=()):
    records = valid_records() if records is None else records
    attrs = (
        ("cim.placement_policy", f'"{oracle.PLACEMENT_POLICY}"'),
        ("cim.route_policy", f'"{oracle.ROUTE_POLICY}"'),
        ("cim.target_profile", f'"{oracle.PROFILE_ID}"'),
        ("cim.target_profile_version", f"{oracle.PROFILE_VERSION} : i64"),
    )
    function_attrs = ", ".join(
        f"{name} = {value}" for name, value in attrs if name not in omit_attrs)
    lines = [f"func.func @mapped() attributes {{{function_attrs}}} {{"]
    for record in records:
        array = lambda values: "array<i64: " + ", ".join(map(str, values)) + ">"
        mapping = (
            f"core_coord = {array(record.core_coord)}, "
            f"destination = {array(record.destination)}, "
            f"ingress = {array(record.ingress)}, route = {array(record.route)}, "
            f"source = {array(record.source)}")
        lines.append(
            f"  %v{record.work_id} = cim.vmm %input, %weight "
            f"{{cim.mapping = {{{mapping}}}, core_slot = {record.core_slot} : i64, "
            f"macro_slot = {record.macro_slot} : i64, work_id = {record.work_id} : i64}} "
            ": tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>")
    lines.append("  return")
    lines.append("}")
    return "\n".join(lines)


class RouteOracleTest(unittest.TestCase):
    def test_all_20_cores_zero_and_corner(self):
        records = oracle.validate_dump(render_dump(), require_all_slots=True)
        self.assertEqual(len(records), 40)
        self.assertEqual({record.core_coord for record in records}, set(oracle.PROFILE_SLOTS))
        self.assertEqual(oracle.shortest_distance((0, 0), (0, 0)), 0)
        self.assertEqual(oracle.shortest_distance((0, 0), (3, 4)), 4)
        self.assertEqual(records[0].route, (0, 0, 0, 0, 0, 0))
        self.assertEqual(records[38].route, (3, 1, 0, 0, 0, 0))

    def test_deterministic_parse_and_validation(self):
        text = render_dump()
        first = oracle.validate_dump(text, require_all_slots=True)
        second = oracle.validate_dump(text, require_all_slots=True)
        self.assertEqual(first, second)
        self.assertEqual(render_dump(first), render_dump(second))

    def test_missing_profile_version_and_policies(self):
        attrs = (
            "cim.target_profile", "cim.target_profile_version",
            "cim.placement_policy", "cim.route_policy",
        )
        for attr in attrs:
            with self.subTest(attr=attr), self.assertRaisesRegex(
                    oracle.OracleError, f"missing function attribute {attr}"):
                oracle.validate_dump(render_dump(omit_attrs={attr}))

    def test_mapping_faults(self):
        records = valid_records()
        faults = []

        changed = list(records)
        changed[0] = changed[0]._replace(core_slot=20)
        faults.append(("invalid slot", tuple(changed), "invalid core_slot 20"))

        changed = list(records)
        changed[0] = changed[0]._replace(macro_slot=2)
        faults.append(("invalid macro", tuple(changed), "invalid macro_slot 2"))

        changed = list(records)
        changed[0] = changed[0]._replace(route=(0, 0, 0, 1, 0, 0))
        faults.append(("nonzero Copy", tuple(changed), "nonzero Copy route"))

        changed = list(records)
        changed[38] = changed[38]._replace(route=(3, 0, 0, 0, 0, 0))
        faults.append(("route mismatch", tuple(changed), "route endpoint"))

        changed = list(records)
        changed[38] = changed[38]._replace(
            ingress=oracle.UPPER_RIGHT, source=oracle.UPPER_RIGHT,
            route=(0, 0, 0, 0, 0, 0))
        faults.append(("upper-right source", tuple(changed), "unsupported ingress/source"))

        changed = list(records)
        changed[38] = changed[38]._replace(core_coord=(3, 3))
        faults.append(("stale mapping", tuple(changed), "stale core_coord"))

        faults.append(("missing work", records[:10] + records[11:], "missing or non-dense work_id"))

        for name, faulty_records, error in faults:
            with self.subTest(fault=name), self.assertRaisesRegex(
                    oracle.OracleError, error):
                oracle.validate_dump(render_dump(faulty_records))

        partial = re.sub(r", route = array<i64: [^>]+>", "", render_dump(), count=1)
        with self.assertRaisesRegex(oracle.OracleError, "incomplete cim.mapping: missing route"):
            oracle.validate_dump(partial)


if __name__ == "__main__":
    unittest.main(
        testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
