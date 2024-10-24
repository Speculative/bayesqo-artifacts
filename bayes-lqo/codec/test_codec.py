from unittest import TestCase

from hypothesis import event, given, settings
from hypothesis.strategies import integers, lists

from .codec import (
    AliasesCodec,
    HashProbeStackMachineCodec,
    JoinOperator,
    JoinTreeBranch,
    JoinTreeLeaf,
    LinearProbeStackMachineCodec,
    StackMachineCodec,
    StackMachineWithOperatorsCodec,
)


def single_alias_tables(tables: list[str]):
    return [(table, 1) for table in tables]


class TestCodec(TestCase):
    def setUp(self):
        self.tables = ["a", "b", "c", "d"]
        self.codec = StackMachineCodec(self.tables)

    def test_left_deep(self):
        left_deep = JoinTreeBranch(
            JoinTreeBranch(
                JoinTreeBranch(JoinTreeLeaf("a"), JoinTreeLeaf("b"), op=None),
                JoinTreeLeaf("c"),
                op=None,
            ),
            JoinTreeLeaf("d"),
            op=None,
        )
        # left_deep.visualize()
        encoded = self.codec.encode(left_deep)
        decoded = self.codec.decode(single_alias_tables(self.tables), encoded)
        # decoded.visualize()
        assert left_deep.equal(decoded)
        # print("left_deep:", left_deep.to_join_clause())

    def test_right_deep(self):
        right_deep = JoinTreeBranch(
            JoinTreeLeaf("a"),
            JoinTreeBranch(
                JoinTreeLeaf("b"),
                JoinTreeBranch(JoinTreeLeaf("c"), JoinTreeLeaf("d"), op=None),
                op=None,
            ),
            op=None,
        )
        # right_deep.visualize()
        encoded = self.codec.encode(right_deep)
        decoded = self.codec.decode(single_alias_tables(self.tables), encoded)
        # decoded.visualize()
        assert right_deep.equal(decoded)
        # print("right_deep:", right_deep.to_join_clause())

    def test_bushy(self):
        bushy = JoinTreeBranch(
            JoinTreeBranch(JoinTreeLeaf("a"), JoinTreeLeaf("b"), op=None),
            JoinTreeBranch(JoinTreeLeaf("c"), JoinTreeLeaf("d"), op=None),
            op=None,
        )
        # bushy.visualize()
        encoded = self.codec.encode(bushy)
        decoded = self.codec.decode(single_alias_tables(self.tables), encoded)
        # decoded.visualize()
        assert bushy.equal(decoded)
        # print("bushy:", bushy.to_join_clause())

    def test_zigzag(self):
        zigzag = JoinTreeBranch(
            JoinTreeLeaf("d"),
            JoinTreeBranch(
                JoinTreeBranch(JoinTreeLeaf("a"), JoinTreeLeaf("b"), op=None),
                JoinTreeLeaf("c"),
                op=None,
            ),
            op=None,
        )
        # zigzag.visualize()
        encoded = self.codec.encode(zigzag)
        decoded = self.codec.decode(single_alias_tables(self.tables), encoded)
        # decoded.visualize()
        assert zigzag.equal(decoded)
        # print("zigzag:", zigzag.to_join_clause())

    def test_zigzag_2(self):
        zigzag2 = JoinTreeBranch(
            JoinTreeBranch(
                JoinTreeLeaf("c"),
                JoinTreeBranch(JoinTreeLeaf("a"), JoinTreeLeaf("b"), op=None),
                op=None,
            ),
            JoinTreeLeaf("d"),
            op=None,
        )
        # zigzag2.visualize()
        encoded = self.codec.encode(zigzag2)
        decoded = self.codec.decode(single_alias_tables(self.tables), encoded)
        # decoded.visualize()
        assert zigzag2.equal(decoded)
        # print("zigzag2:", zigzag2.to_join_clause())


class TestLinearProbeCodec(TestCase):
    def test_left_deep(self):
        tables = ["a", "b", "c", "d"]
        codec = LinearProbeStackMachineCodec(tables)

        left_deep = JoinTreeBranch(
            JoinTreeBranch(
                JoinTreeBranch(JoinTreeLeaf("a"), JoinTreeLeaf("b"), op=None),
                JoinTreeLeaf("c"),
                op=None,
            ),
            JoinTreeLeaf("d"),
            op=None,
        )
        # left_deep.visualize()
        encoded = codec.encode(left_deep)
        decoded = codec.decode(single_alias_tables(tables), encoded)
        # decoded.visualize()
        assert left_deep.equal(decoded)

    def test_decode_repeat(self):
        codec = LinearProbeStackMachineCodec(["a", "b", "c", "d"])
        decoded = codec.decode(single_alias_tables(["a", "b", "c", "d"]), [0, 1, 0, 0, 0, 0])
        tables = decoded.tables()
        for table in ["a", "b", "c", "d"]:
            assert table in tables

    def test_decode_unknown_symbols(self):
        codec = LinearProbeStackMachineCodec(["a", "b", "c", "d"])
        decoded = codec.decode(single_alias_tables(["a", "b", "c", "d"]), [0, 1, 10, 16, 0, 3])
        tables = decoded.tables()
        assert len(tables) == 4
        for table in ["a", "b", "c", "d"]:
            assert table in tables

    def test_decode_too_short(self):
        codec = LinearProbeStackMachineCodec(["a", "b", "c", "d"])
        decoded = codec.decode(single_alias_tables(["a", "b", "c", "d"]), [0, 1])
        tables = decoded.tables()
        assert len(tables) == 4
        for table in ["a", "b", "c", "d"]:
            assert table in tables

    def test_decode_too_long(self):
        codec = LinearProbeStackMachineCodec(["a", "b", "c", "d"])
        decoded = codec.decode(single_alias_tables(["a", "b", "c", "d"]), [0, 1, 0, 2, 0, 3, 0, 4])
        tables = decoded.tables()
        assert len(tables) == 4
        for table in ["a", "b", "c", "d"]:
            assert table in tables

    def test_decode_non_query_symbols(self):
        codec = LinearProbeStackMachineCodec(["a", "b", "c", "d", "e", "f"])
        decoded = codec.decode(single_alias_tables(["a", "b", "e", "f"]), [0, 1, 0, 2, 0, 3])
        tables = decoded.tables()
        assert len(tables) == 4
        for table in ["a", "b", "e", "f"]:
            assert table in tables


class TestHashProbeCodec(TestCase):
    def test_left_deep(self):
        tables = ["a", "b", "c", "d"]
        codec = HashProbeStackMachineCodec(tables)

        left_deep = JoinTreeBranch(
            JoinTreeBranch(
                JoinTreeBranch(JoinTreeLeaf("a"), JoinTreeLeaf("b"), op=None),
                JoinTreeLeaf("c"),
                op=None,
            ),
            JoinTreeLeaf("d"),
            op=None,
        )
        # left_deep.visualize()
        encoded = codec.encode(left_deep)
        decoded = codec.decode(single_alias_tables(tables), encoded)
        # decoded.visualize()
        assert left_deep.equal(decoded)

    def test_decode_repeat(self):
        codec = HashProbeStackMachineCodec(["a", "b", "c", "d"])
        decoded = codec.decode(single_alias_tables(["a", "b", "c", "d"]), [0, 1, 0, 0, 0, 0])
        tables = decoded.tables()
        for table in ["a", "b", "c", "d"]:
            assert table in tables

    def test_decode_unknown_symbols(self):
        codec = HashProbeStackMachineCodec(["a", "b", "c", "d"])
        decoded = codec.decode(single_alias_tables(["a", "b", "c", "d"]), [0, 1, 10, 16, 0, 3])
        tables = decoded.tables()
        assert len(tables) == 4
        for table in ["a", "b", "c", "d"]:
            assert table in tables

    def test_decode_too_short(self):
        codec = HashProbeStackMachineCodec(["a", "b", "c", "d"])
        decoded = codec.decode(single_alias_tables(["a", "b", "c", "d"]), [0, 1])
        tables = decoded.tables()
        assert len(tables) == 4
        for table in ["a", "b", "c", "d"]:
            assert table in tables

    def test_decode_too_long(self):
        codec = HashProbeStackMachineCodec(["a", "b", "c", "d"])
        decoded = codec.decode(single_alias_tables(["a", "b", "c", "d"]), [0, 1, 0, 2, 0, 3, 0, 4])
        tables = decoded.tables()
        assert len(tables) == 4
        for table in ["a", "b", "c", "d"]:
            assert table in tables

    def test_decode_non_query_symbols(self):
        codec = HashProbeStackMachineCodec(["a", "b", "c", "d", "e", "f"])
        decoded = codec.decode(single_alias_tables(["a", "b", "e", "f"]), [0, 1, 0, 2, 0, 3])
        tables = decoded.tables()
        assert len(tables) == 4
        for table in ["a", "b", "e", "f"]:
            assert table in tables


class TestHashProbeEncodeDecode(TestCase):
    @settings(max_examples=1_000)
    @given(
        encoded_1=lists(integers(min_value=0, max_value=5), min_size=1),
        encoded_2=lists(integers(min_value=0, max_value=5), min_size=1),
    )
    def test_tree_equal_hash_equal(self, encoded_1: list[int], encoded_2: list[int]):
        codec = HashProbeStackMachineCodec(["a", "b", "c", "d", "e", "f"])
        decoded_1 = codec.decode(single_alias_tables(["a", "b", "e", "f"]), encoded_1)
        decoded_2 = codec.decode(single_alias_tables(["a", "b", "e", "f"]), encoded_2)
        assert decoded_1.equal(decoded_2) == (decoded_1.stable_hash() == decoded_2.stable_hash())
        for table in ["a", "b", "e", "f"]:
            assert table in decoded_1.tables()


class TestOperatorCodec(TestCase):
    def setUp(self):
        self.tables = ["a", "b", "c", "d"]
        self.codec = StackMachineWithOperatorsCodec(self.tables)

    def test_left_deep(self):
        left_deep = JoinTreeBranch(
            JoinTreeBranch(
                JoinTreeBranch(JoinTreeLeaf("a"), JoinTreeLeaf("b"), JoinOperator.HashJoin),
                JoinTreeLeaf("c"),
                JoinOperator.NestLoop,
            ),
            JoinTreeLeaf("d"),
            JoinOperator.MergeJoin,
        )
        encoded = self.codec.encode(left_deep)
        decoded = self.codec.decode(single_alias_tables(self.tables), encoded)
        assert left_deep.equal(decoded)

    def test_right_deep(self):
        right_deep = JoinTreeBranch(
            JoinTreeLeaf("a"),
            JoinTreeBranch(
                JoinTreeLeaf("b"),
                JoinTreeBranch(JoinTreeLeaf("c"), JoinTreeLeaf("d"), JoinOperator.HashJoin),
                JoinOperator.HashJoin,
            ),
            JoinOperator.NestLoop,
        )
        encoded = self.codec.encode(right_deep)
        decoded = self.codec.decode(single_alias_tables(self.tables), encoded)
        assert right_deep.equal(decoded)

    def test_bushy(self):
        bushy = JoinTreeBranch(
            JoinTreeBranch(JoinTreeLeaf("a"), JoinTreeLeaf("b"), JoinOperator.NestLoop),
            JoinTreeBranch(JoinTreeLeaf("c"), JoinTreeLeaf("d"), JoinOperator.NestLoop),
            JoinOperator.MergeJoin,
        )
        encoded = self.codec.encode(bushy)
        decoded = self.codec.decode(single_alias_tables(self.tables), encoded)
        assert bushy.equal(decoded)

    def test_zigzag(self):
        zigzag = JoinTreeBranch(
            JoinTreeLeaf("d"),
            JoinTreeBranch(
                JoinTreeBranch(JoinTreeLeaf("a"), JoinTreeLeaf("b"), JoinOperator.HashJoin),
                JoinTreeLeaf("c"),
                JoinOperator.HashJoin,
            ),
            JoinOperator.HashJoin,
        )
        encoded = self.codec.encode(zigzag)
        decoded = self.codec.decode(single_alias_tables(self.tables), encoded)
        assert zigzag.equal(decoded)

    def test_zigzag_2(self):
        zigzag2 = JoinTreeBranch(
            JoinTreeBranch(
                JoinTreeLeaf("c"),
                JoinTreeBranch(JoinTreeLeaf("a"), JoinTreeLeaf("b"), JoinOperator.MergeJoin),
                JoinOperator.MergeJoin,
            ),
            JoinTreeLeaf("d"),
            JoinOperator.MergeJoin,
        )
        encoded = self.codec.encode(zigzag2)
        decoded = self.codec.decode(single_alias_tables(self.tables), encoded)
        assert zigzag2.equal(decoded)


class TestOperatorHashProbeEncodeDecode(TestCase):
    @settings(max_examples=1_000)
    @given(
        encoded_1=lists(integers(min_value=0, max_value=5), min_size=1),
        encoded_2=lists(integers(min_value=0, max_value=5), min_size=1),
    )
    def test_operator_tree_equal_hash_equal(self, encoded_1: list[int], encoded_2: list[int]):
        codec = HashProbeStackMachineCodec(["a", "b", "c", "d", "e", "f"])
        decoded_1 = codec.decode(single_alias_tables(["a", "b", "e", "f"]), encoded_1)
        decoded_2 = codec.decode(single_alias_tables(["a", "b", "e", "f"]), encoded_2)
        assert decoded_1.equal(decoded_2) == (decoded_1.stable_hash() == decoded_2.stable_hash())
        for table in ["a", "b", "e", "f"]:
            assert table in decoded_1.tables()


class TestAliasCodec(TestCase):
    def setUp(self):
        self.tables = ["a", "b", "c", "d"]
        self.codec = AliasesCodec(self.tables)

    def test_left_deep_aliases(self):
        tree = JoinTreeBranch(
            JoinTreeBranch(
                JoinTreeBranch(JoinTreeLeaf("a", 1), JoinTreeLeaf("b", 1), op=JoinOperator.HashJoin),
                JoinTreeLeaf("a", 2),
                op=JoinOperator.HashJoin,
            ),
            JoinTreeLeaf("d", 1),
            op=JoinOperator.HashJoin,
        )
        encoded = self.codec.encode(tree)
        decoded = self.codec.decode([("a", 2), ("b", 1), ("d", 1)], encoded)
        # tree.visualize()
        # decoded.visualize()
        assert tree.equal(decoded)

    @settings(max_examples=1_000)
    @given(
        encoded=lists(integers(min_value=0, max_value=5), min_size=1),
    )
    def test_alias_codec_decode(self, encoded: list[int]):
        codec = AliasesCodec(["a", "b", "c", "d", "e", "f"])
        decoded = codec.decode([("a", 3), ("b", 1), ("e", 2), ("f", 1)], encoded)

    @settings(max_examples=1_000)
    @given(
        encoded_1=lists(integers(min_value=0, max_value=5), min_size=1),
    )
    def test_alias_codec_encode_decode_roundtrip(self, encoded_1: list[int]):
        all_tables = ["a", "b", "c", "d", "e", "f"]
        query_alias_counts = [("a", 3), ("b", 1), ("e", 2), ("f", 1)]

        codec = AliasesCodec(all_tables)
        decoded_1 = codec.decode(query_alias_counts, encoded_1)
        encoded_2 = codec.encode(decoded_1)
        decoded_2 = codec.decode(query_alias_counts, encoded_2)

        assert decoded_1.equal(decoded_2)
        assert decoded_1.stable_hash() == decoded_2.stable_hash()

    @settings(max_examples=1_000)
    @given(
        encoded_1=lists(integers(min_value=0, max_value=5), min_size=1),
        encoded_2=lists(integers(min_value=0, max_value=5), min_size=1),
    )
    def test_alias_tree_equal_hash_equal(self, encoded_1: list[int], encoded_2: list[int]):
        all_tables = ["a", "b", "c", "d", "e", "f"]
        query_alias_counts = [("a", 3), ("b", 1), ("e", 2), ("f", 1)]

        codec = AliasesCodec(all_tables)
        decoded_1 = codec.decode(query_alias_counts, encoded_1)
        decoded_2 = codec.decode(query_alias_counts, encoded_2)
        event("join trees equal", str(decoded_1.equal(decoded_2)))
        assert decoded_1.equal(decoded_2) == (decoded_1.stable_hash() == decoded_2.stable_hash())
        for table in ["a", "b", "e", "f"]:
            assert table in decoded_1.tables()
