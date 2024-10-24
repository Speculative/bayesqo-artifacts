import hashlib
import io
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pydot  # type: ignore


class JoinOperator(Enum):
    NestLoop = 1
    HashJoin = 2
    MergeJoin = 3


class JoinTree(ABC):
    @abstractmethod
    def clone(self) -> "JoinTree":
        pass

    @abstractmethod
    def is_left_deep(self) -> bool:
        pass

    @abstractmethod
    def is_right_deep(self) -> bool:
        pass

    @abstractmethod
    def is_zigzag(self) -> bool:
        pass

    def is_bushy(self) -> bool:
        return not (self.is_left_deep() or self.is_right_deep() or self.is_zigzag())

    @abstractmethod
    def height(self) -> int:
        pass

    @abstractmethod
    def equal(self, other: "JoinTree") -> bool:
        pass

    @abstractmethod
    def structurally_equal(self, other: "JoinTree") -> bool:
        pass

    @abstractmethod
    def stable_hash(self) -> str:
        pass

    @abstractmethod
    def tables(self) -> set[str]:
        pass

    @abstractmethod
    def tables_aliases(self) -> set[tuple[str, int]]:
        pass

    @abstractmethod
    def aliases(self) -> set[str]:
        pass

    @abstractmethod
    def visualize(self, save_to: Optional[str] = None):
        pass

    @abstractmethod
    def _visualize(self, graph: pydot.Dot, id: int, parent_id) -> int:
        pass

    @abstractmethod
    def to_join_clause(self) -> str:
        pass

    @abstractmethod
    def to_operator_hint(self) -> str:
        pass

    @abstractmethod
    def all_branches(self) -> set["JoinTreeBranch"]:
        pass

    @abstractmethod
    def all_leaves(self) -> set["JoinTreeLeaf"]:
        pass


class JoinTreeBranch(JoinTree):
    left: JoinTree
    right: JoinTree
    op: Optional[JoinOperator]

    __match_args__ = ("left", "right", "op")

    def __init__(self, left: JoinTree, right: JoinTree, op: Optional[JoinOperator]):
        self.left = left
        self.right = right
        self.op = op

    def clone(self) -> "JoinTreeBranch":
        left = self.left.clone()
        right = self.right.clone()
        return JoinTreeBranch(left, right, self.op)

    def is_left_deep(self):
        return isinstance(self.right, JoinTreeLeaf) and self.left.is_left_deep()

    def is_right_deep(self):
        return isinstance(self.left, JoinTreeLeaf) and self.right.is_right_deep()

    def is_zigzag(self):
        if isinstance(self.left, JoinTreeLeaf):
            return self.right.is_zigzag()
        elif isinstance(self.right, JoinTreeLeaf):
            return self.left.is_zigzag()
        else:
            return False

    def height(self):
        return 1 + max(self.left.height(), self.right.height())

    def tables(self) -> set[str]:
        return self.left.tables().union(self.right.tables())

    def tables_aliases(self) -> set[tuple[str, int]]:
        return self.left.tables_aliases().union(self.right.tables_aliases())

    def aliases(self) -> set[str]:
        return self.left.aliases().union(self.right.aliases())

    def equal(self, other: JoinTree) -> bool:
        match other:
            case JoinTreeLeaf(_):
                return False
            case JoinTreeBranch(other_left, other_right, other_op):
                return self.left.equal(other_left) and self.right.equal(other_right) and self.op == other_op
        raise ValueError("JoinTree should be branch or leaf")

    def structurally_equal(self, other: JoinTree) -> bool:
        match other:
            case JoinTreeLeaf(_):
                return False
            case JoinTreeBranch(other_left, other_right, _):
                return (
                    self.left.structurally_equal(other_left)
                    and self.right.structurally_equal(other_right)
                    and self.op == other.op
                )
        raise ValueError("JoinTree should be branch or leaf")

    def stable_hash(self) -> str:
        h = hashlib.md5()
        h.update(self.left.stable_hash().encode("utf-8"))
        h.update(self.right.stable_hash().encode("utf-8"))
        if self.op is not None:
            h.update(self.op.name.encode("utf-8"))
        return h.hexdigest()

    def __hash__(self):
        return hash((self.left, self.right))

    def __eq__(self, other) -> bool:
        return isinstance(other, JoinTreeBranch) and self.equal(other)

    def visualize(self, save_to: Optional[str] = None):
        graph = pydot.Dot()
        graph.add_node(pydot.Node(0, label=self.op.name if self.op else "⨝"))
        id_base = self.left._visualize(graph, 1, 0)
        self.right._visualize(graph, id_base + 1, 0)

        if save_to is not None:
            if save_to[-4:] == ".svg":
                svg_str = graph.create_svg(prog="dot")
                with open(save_to, "wb") as f:
                    f.write(svg_str)
                return
            elif save_to[-4:] == ".png":
                png_str = graph.create_png(prog="dot")
                with open(save_to, "wb") as f:
                    f.write(png_str)
                return

        # dot = nx.nx_pydot.to_pydot(graph)
        # render the `pydot` by calling `dot`, no file saved to disk
        png_str = graph.create_png(prog="dot")

        # treat the DOT output as an image file
        sio = io.BytesIO()
        sio.write(png_str)
        sio.seek(0)
        img = mpimg.imread(sio)
        imgplot = plt.imshow(img, aspect="equal")
        plt.axis("off")
        if save_to is not None:
            plt.savefig(save_to)
        else:
            plt.show()

    def _visualize(self, graph: pydot.Dot, id: int, parent_id) -> int:
        label = id
        graph.add_node(pydot.Node(label, label=self.op.name if self.op else "⨝"))
        graph.add_edge(pydot.Edge(parent_id, label))
        id_base = id

        id_base += self.left._visualize(graph, id_base + 1, label)
        id_base += self.right._visualize(graph, id_base + 1, label)

        return id_base

    def to_join_clause(self) -> str:
        left = self.left.to_join_clause()
        right = self.right.to_join_clause()

        # We can't fully parenthesize because Postgres doesn't like it when single tables are parenthesized
        # So we only add parentheses around non-leaf nodes
        match (self.left, self.right):
            case JoinTreeLeaf(), JoinTreeLeaf():
                return f"{left} CROSS JOIN {right}"
            case JoinTreeLeaf(), JoinTreeBranch(_, _, _):
                return f"{left} CROSS JOIN ({right})"
            case JoinTreeBranch(_, _, _), JoinTreeLeaf():
                return f"({left}) CROSS JOIN {right}"
            case JoinTreeBranch(_, _, _), JoinTreeBranch(_, _, _):
                return f"({left}) CROSS JOIN ({right})"

        raise ValueError("JoinTree should be branch or leaf")

    def to_operator_hint(self) -> str:
        if self.op is None:
            raise ValueError("Cannot get operator hint for JoinTreeBranch without op")

        left_hint = self.left.to_operator_hint()
        if left_hint:
            left_hint = f"{left_hint}\n"
        right_hint = self.right.to_operator_hint()
        if right_hint:
            right_hint = f"{right_hint}\n"

        return left_hint + right_hint + (f"{self.op.name}({' '.join(self.aliases())})")

    def all_branches(self):
        return set([self]) | self.left.all_branches() | self.right.all_branches()

    def all_leaves(self):
        return self.left.all_leaves() | self.right.all_leaves()


class JoinTreeLeaf(JoinTree):
    table: str
    alias: Optional[int]

    __match_args__ = ("table", "alias")

    def __init__(self, table: str, alias: Optional[int] = None):
        self.table = table
        self.alias = alias

    def clone(self) -> "JoinTreeLeaf":
        return JoinTreeLeaf(self.table, self.alias)

    def is_left_deep(self):
        return True

    def is_right_deep(self):
        return True

    def is_zigzag(self):
        return True

    def height(self):
        return 1

    def structurally_equal(self, other: JoinTree) -> bool:
        match other:
            case JoinTreeLeaf(other_table, other_alias):
                return self.table == other_table
            case JoinTreeBranch(_, _, _):
                return False
        raise ValueError("JoinTree should be branch or leaf")

    def equal(self, other: JoinTree) -> bool:
        match other:
            case JoinTreeLeaf(other_table, other_alias):
                return self.table == other_table and self.alias == other_alias
            case JoinTreeBranch(_, _, _):
                return False
        raise ValueError("JoinTree should be branch or leaf")

    def stable_hash(self) -> str:
        h = hashlib.md5()
        h.update(self.table.encode("utf-8"))
        if self.alias is not None:
            h.update(str(self.alias).encode("utf-8"))
        return h.hexdigest()

    def __hash__(self):
        if self.alias is not None:
            return hash((self.table, self.alias))
        return hash(self.table)

    def __eq__(self, other):
        return isinstance(other, JoinTreeLeaf) and self.table == other.table and self.alias == other.alias

    def tables(self) -> set[str]:
        return set([self.table])

    def tables_aliases(self) -> set[tuple[str, int]]:
        if self.alias is None:
            raise ValueError("JoinTree node is missing alias")
        return set([(self.table, self.alias)])

    def aliases(self) -> set[str]:
        if self.alias is None:
            return self.tables()
        return set([f"{self.table}{self.alias}"])

    def visualize(self, save_to: Optional[str] = None):
        raise ValueError("Cannot visualize just a leaf node")

    def _visualize(self, graph: pydot.Dot, id: int, parent_id) -> int:
        label = f"{self.table},  {self.alias}" if self.alias is not None else self.table
        graph.add_node(pydot.Node(label))
        graph.add_edge(pydot.Edge(parent_id, label))
        id_base = id

        return id_base

    def to_join_clause(self) -> str:
        if self.alias is None:
            return f"{self.table} AS {self.table}1"
        return f"{self.table} AS {self.table}{self.alias}"

    def to_operator_hint(self) -> str:
        return ""

    def all_branches(self) -> set[JoinTreeBranch]:
        return set()

    def all_leaves(self) -> set["JoinTreeLeaf"]:
        return set([self])


class SymbolTable:
    """
    Immutable table mapping symbols (ints) to intermediate join trees,
    as well as all other symbols representing the same join tree
    """

    __all_symbols: list[str]
    __query: list[str]

    symbol_to_tree: dict[int, JoinTree]
    tree_to_symbols: dict[JoinTree, set[int]]

    def __init__(self, all_symbols: list[str], query: list[str]):
        self.__all_symbols = all_symbols
        self.__query = query

        self.symbol_to_tree = {}
        self.tree_to_symbols = {}

        symbol_index = {table: index for index, table in enumerate(all_symbols)}
        for table in query:
            symbol = symbol_index[table]
            leaf = JoinTreeLeaf(table)
            self.symbol_to_tree[symbol] = leaf
            self.tree_to_symbols[leaf] = set([symbol])

    def get(self, symbol: int) -> Optional[JoinTree]:
        """If a symbol has never been set (i.e. not in the query), will return None"""
        return self.symbol_to_tree[symbol] if symbol in self.symbol_to_tree else None

    def clone(self) -> "SymbolTable":
        clone = SymbolTable(self.__all_symbols.copy(), self.__query.copy())
        clone.symbol_to_tree = self.symbol_to_tree.copy()
        clone.tree_to_symbols = {tree: symbol_list.copy() for tree, symbol_list in self.tree_to_symbols.items()}
        return clone

    def with_join(self, old_left: JoinTree, old_right: JoinTree, new: JoinTree):
        clone = self.clone()
        existing_symbols = clone.tree_to_symbols[old_left].union(clone.tree_to_symbols[old_right])
        for symbol in existing_symbols:
            clone.symbol_to_tree[symbol] = new

        clone.tree_to_symbols[new] = existing_symbols
        # Addressing the symbol table with the old tree is now invalid
        del clone.tree_to_symbols[old_left]
        del clone.tree_to_symbols[old_right]

        return clone

    def without(self, tree: JoinTree):
        clone = self.clone()
        for symbol in clone.tree_to_symbols[tree]:
            del clone.symbol_to_tree[symbol]
        del clone.tree_to_symbols[tree]
        return clone

    def query_symbols(self):
        return {
            table: self.__all_symbols.index(table)
            for table in self.__query
            if self.__all_symbols.index(table) in self.symbol_to_tree
        }

    def all_symbols(self):
        return self.__all_symbols


class SymbolResolver(ABC):
    @abstractmethod
    def resolve_symbol(self, symbol_table: SymbolTable, symbol: int) -> JoinTree:
        pass


class NullSymbolResolver(SymbolResolver):
    def resolve_symbol(self, symbol_table: SymbolTable, symbol: int) -> JoinTree:
        tree = symbol_table.get(symbol)
        if tree is None:
            raise ValueError("NullSymbolResolver only accepts perfect encodes")
        return tree


class LinearProbeSymbolResolver(SymbolResolver):
    def resolve_symbol(self, symbol_table: SymbolTable, symbol: int) -> JoinTree:
        total_symbols = len(symbol_table.all_symbols())

        count = 0
        while (tree := symbol_table.get(symbol)) is None:
            symbol = (symbol + 1) % total_symbols

            count += 1
            if count > total_symbols:
                raise ValueError("LinearProbeStackMachineCodec failed to resolve symbol")

        return tree


class HashProbeSymbolResolver(SymbolResolver):
    def resolve_symbol(self, symbol_table: SymbolTable, symbol: int) -> JoinTree:
        tree = symbol_table.get(symbol)
        if tree is not None:
            return tree

        query_symbols = [symbol for symbol in sorted(symbol_table.query_symbols().values())]
        symbol = query_symbols[hash(symbol) % len(query_symbols)]
        tree = symbol_table.get(symbol)
        if tree is not None:
            return tree

        raise ValueError("HashProbeStackMachineCodec somehow failed to resolve symbol")


class Codec(ABC):
    @abstractmethod
    def encode(self, tree: JoinTree) -> list[int]:
        pass

    @abstractmethod
    def decode(self, query: list[tuple[str, int]], encoded: list[int]) -> JoinTree:
        pass


class NonAliasCodec(Codec):
    # An enumeration of and ordering on all tables in the schema
    base_table_symbols: list[str]
    symbol_resolver: SymbolResolver

    def __init__(
        self,
        base_table_symbols: list[str],
        symbol_resolver: SymbolResolver = NullSymbolResolver(),
    ):
        self.base_table_symbols = base_table_symbols
        self.symbol_resolver = symbol_resolver

    @abstractmethod
    def encode(self, tree: JoinTree) -> list[int]:
        pass

    @abstractmethod
    def decode(self, query: list[tuple[str, int]], encoded: list[int]) -> JoinTree:
        pass


class StackMachineCodec(NonAliasCodec):
    def encode(self, tree: JoinTree) -> list[int]:
        # Walk over tree, get all internal nodes
        remaining = tree.all_branches()
        # JoinTree -> set of symbols representing the join
        encoded_nodes: dict[JoinTree, list[int]] = {
            node: [self.base_table_symbols.index(node.table)] for node in tree.all_leaves()
        }
        out: list[int] = []
        while remaining:
            # Need to copy the set because we mutate it while iterating
            for candidate in remaining:
                if candidate.left in encoded_nodes and candidate.right in encoded_nodes:
                    # internal_node.visualize()
                    left_symbols = encoded_nodes[candidate.left]
                    right_symbols = encoded_nodes[candidate.right]
                    encoded_nodes[candidate] = left_symbols + right_symbols
                    out += [left_symbols[0], right_symbols[0]]

                    remaining = {node for node in remaining if node is not candidate}
                    break
                    # remaining.remove(internal_node)

        return out

    def decode(self, query: list[tuple[str, int]], encoded: list[int]) -> JoinTree:
        if any(alias_number > 1 for _, alias_number in query):
            raise ValueError("JoinOrder language cannot handle aliases")
        query: list[str] = [table for table, _ in query]

        needed_tables = set(query)
        symbol_table = SymbolTable(self.base_table_symbols, query)

        # In case we get handed completely empty input somehow
        if encoded == []:
            encoded = [0, 1]

        if len(encoded) % 2 != 0:
            encoded.append(0)

        joins = [encoded[i : i + 2] for i in range(0, len(encoded), 2)]
        for join in joins:
            left_symbol, right_symbol = join
            left_tree = self.symbol_resolver.resolve_symbol(symbol_table, left_symbol)
            right_tree = self.symbol_resolver.resolve_symbol(symbol_table.without(left_tree), right_symbol)
            new_tree = JoinTreeBranch(left_tree, right_tree, op=None)
            symbol_table = symbol_table.with_join(left_tree, right_tree, new_tree)

            # In case of finishing early (input is longer than necessary)
            if all(table in new_tree.tables() for table in needed_tables):
                break

        # Just add any remaining tables as a left deep tree
        included_tables = set(leaf.table for leaf in new_tree.all_leaves())
        for table in sorted(needed_tables - included_tables):
            new_tree = JoinTreeBranch(new_tree, JoinTreeLeaf(table), op=None)

        return new_tree


class LinearProbeStackMachineCodec(StackMachineCodec):
    def __init__(self, base_table_symbols: list[str]):
        super().__init__(base_table_symbols, LinearProbeSymbolResolver())


class HashProbeStackMachineCodec(StackMachineCodec):
    def __init__(self, base_table_symbols: list[str]):
        super().__init__(base_table_symbols, HashProbeSymbolResolver())


class StackMachineWithOperatorsCodec(NonAliasCodec):
    def encode(self, tree: JoinTree) -> list[int]:
        # Walk over tree, get all internal nodes
        remaining = tree.all_branches()
        # JoinTree -> set of symbols representing the join
        encoded_nodes: dict[JoinTree, list[int]] = {
            node: [self.base_table_symbols.index(node.table)] for node in tree.all_leaves()
        }
        out: list[int] = []
        while remaining:
            for candidate in remaining:
                if candidate.left in encoded_nodes and candidate.right in encoded_nodes:
                    # internal_node.visualize()
                    left_symbols = encoded_nodes[candidate.left]
                    right_symbols = encoded_nodes[candidate.right]
                    encoded_nodes[candidate] = left_symbols + right_symbols
                    if candidate.op is None:
                        raise ValueError("Cannot operator-encode JoinTreeBranch without op")
                    out += [left_symbols[0], right_symbols[0], candidate.op.value]

                    remaining = {node for node in remaining if node is not candidate}
                    break

        return out

    def decode(self, query: list[tuple[str, int]], encoded: list[int]) -> JoinTree:
        if any(alias_number > 1 for _, alias_number in query):
            raise ValueError("JoinOrder language cannot handle aliases")
        query: list[str] = [table for table, _ in query]

        needed_tables = set(query)
        symbol_table = SymbolTable(self.base_table_symbols, query)

        # In case we get handed completely empty input somehow
        if encoded == []:
            encoded = [0, 1, 0]

        # Pad to length 3
        encoded += [0] * ((3 - (len(encoded) % 3)) % 3)

        joins = [encoded[i : i + 3] for i in range(0, len(encoded), 3)]
        for join in joins:
            left_symbol, right_symbol, op_symbol = join
            left_tree = self.symbol_resolver.resolve_symbol(symbol_table, left_symbol)
            right_tree = self.symbol_resolver.resolve_symbol(symbol_table.without(left_tree), right_symbol)

            # This depends on the JoinOperator enum having continuous values from 1.
            # I am only implementing this mod-probe-based resolution. If for some reason we ever
            # want a different resolution we'll need to implement a new type of Resolver.
            try:
                join_op = JoinOperator(op_symbol)
            except ValueError:
                join_op = JoinOperator((op_symbol % len(JoinOperator)) + 1)

            new_tree = JoinTreeBranch(left_tree, right_tree, join_op)
            symbol_table = symbol_table.with_join(left_tree, right_tree, new_tree)

            # In case of finishing early (input is longer than necessary)
            if all(table in new_tree.tables() for table in needed_tables):
                break

        # Just add any remaining tables as a left deep tree
        included_tables = set(leaf.table for leaf in new_tree.all_leaves())
        for table in sorted(needed_tables - included_tables):
            new_tree = JoinTreeBranch(new_tree, JoinTreeLeaf(table), JoinOperator.HashJoin)

        return new_tree


class HashProbeStackMachineWithOperatorsCodec(StackMachineWithOperatorsCodec):
    def __init__(self, base_table_symbols: list[str]):
        super().__init__(base_table_symbols, HashProbeSymbolResolver())


def expand_counts(all_tables: list[str], query_alias_counts: list[tuple[str, int]]):
    return set(
        (all_tables.index(table), alias_num)
        for (table, num_aliases) in query_alias_counts
        for alias_num in range(1, num_aliases + 1)
    )


class AliasSymbolTable:
    # Canonical ordering of int symbols -> table names
    # including tables not in the query
    __all_tables: list[str]

    # All table, alias pairs
    __query_symbols: set[tuple[int, int]]

    symbol_to_tree: dict[tuple[int, int], JoinTree]
    tree_to_symbols: dict[JoinTree, set[tuple[int, int]]]

    def __init__(self, all_tables: list[str], query_symbols: set[tuple[int, int]]):
        self.__all_tables = all_tables
        self.__query_symbols = query_symbols

        self.symbol_to_tree = {}
        self.tree_to_symbols = {}

        for table_symbol, alias_symbol in query_symbols:
            table_name = all_tables[table_symbol]
            symbol = (table_symbol, alias_symbol)
            leaf = JoinTreeLeaf(table_name, alias_symbol)
            self.symbol_to_tree[symbol] = leaf
            self.tree_to_symbols[leaf] = set([symbol])

    def clone(self) -> "AliasSymbolTable":
        clone = AliasSymbolTable(self.__all_tables.copy(), self.__query_symbols.copy())
        clone.symbol_to_tree = self.symbol_to_tree.copy()
        clone.tree_to_symbols = {tree: symbol_list.copy() for tree, symbol_list in self.tree_to_symbols.items()}
        return clone

    def with_join(self, old_left: JoinTree, old_right: JoinTree, new: JoinTree):
        clone = self.clone()
        existing_symbols = clone.tree_to_symbols[old_left].union(clone.tree_to_symbols[old_right])
        for symbol in existing_symbols:
            clone.symbol_to_tree[symbol] = new

        clone.tree_to_symbols[new] = existing_symbols
        # Addressing the symbol table with the old tree is now invalid
        del clone.tree_to_symbols[old_left]
        del clone.tree_to_symbols[old_right]

        return clone

    def without(self, tree: JoinTree):
        clone = self.clone()
        for symbol in clone.tree_to_symbols[tree]:
            del clone.symbol_to_tree[symbol]
        del clone.tree_to_symbols[tree]
        return clone

    def get(self, table_symbol: int, alias_symbol: int) -> Optional[JoinTree]:
        key = (table_symbol, alias_symbol)
        return self.symbol_to_tree[key] if key in self.symbol_to_tree else None

    def get_table_aliases(self, table_symbol: int):
        return [(table, alias_num) for table, alias_num in self.symbol_to_tree.keys() if table == table_symbol]


class AliasesCodec(Codec):
    # An enumeration of and ordering on all tables in the schema
    all_tables: list[str]

    def __init__(self, all_tables: list[str]):
        self.all_tables = all_tables

    def table_symbol_to_name(self, table_symbol: int):
        return self.all_tables[table_symbol]

    def table_name_to_symbol(self, table_name: str):
        return self.all_tables.index(table_name)

    def encode(self, tree: JoinTree) -> list[int]:
        # Walk over tree, get all internal nodes
        remaining = tree.all_branches()
        # JoinTree -> set of symbols representing the join
        encoded_nodes: dict[JoinTree, list[tuple[int, int]]] = {}
        for node in tree.all_leaves():
            if node.alias is None:
                raise ValueError("AliasesCodec can't encode a table without an alias")
            encoded_nodes[node] = [(self.table_name_to_symbol(node.table), node.alias)]
        out: list[int] = []
        while remaining:
            for candidate in sorted(remaining, key=lambda t: min(t.tables())):
                if candidate.left in encoded_nodes and candidate.right in encoded_nodes:
                    left_symbols = encoded_nodes[candidate.left]
                    right_symbols = encoded_nodes[candidate.right]
                    encoded_nodes[candidate] = left_symbols + right_symbols
                    if candidate.op is None:
                        raise ValueError("Cannot operator-encode JoinTreeBranch without op")
                    left_table, left_alias = left_symbols[0]
                    right_table, right_alias = right_symbols[0]
                    out += [
                        left_table,
                        left_alias,
                        right_table,
                        right_alias,
                        candidate.op.value,
                    ]

                    remaining = {node for node in remaining if node is not candidate}
                    break

        return out

    def decode(self, query: list[tuple[str, int]], encoded: list[int]) -> JoinTree:
        query_aliases = expand_counts(self.all_tables, query)
        needed_aliases = set(query_aliases)
        symbol_table = AliasSymbolTable(self.all_tables, query_aliases)

        # In case we get handed completely empty input somehow
        if encoded == []:
            encoded = [0, 0, 1, 0, 0]

        # Pad to length 3
        encoded += [0] * ((5 - (len(encoded) % 5)) % 5)

        new_tree: Optional[JoinTree] = None
        joins = [encoded[i : i + 5] for i in range(0, len(encoded), 5)]
        for join in joins:
            (
                left_table_symbol,
                left_alias_symbol,
                right_table_symbol,
                right_alias_symbol,
                op_symbol,
            ) = join
            left_tree = self.resolve_table_alias(symbol_table, left_table_symbol, left_alias_symbol)
            right_tree = self.resolve_table_alias(
                symbol_table.without(left_tree), right_table_symbol, right_alias_symbol
            )

            # This depends on the JoinOperator enum having continuous values from 1.
            # I am only implementing this mod-probe-based resolution. If for some reason we ever
            # want a different resolution we'll need to implement a new type of Resolver.
            try:
                join_op = JoinOperator(op_symbol)
            except ValueError:
                join_op = JoinOperator((op_symbol % len(JoinOperator)) + 1)

            new_tree = JoinTreeBranch(left_tree, right_tree, join_op)
            symbol_table = symbol_table.with_join(left_tree, right_tree, new_tree)

            # In case of finishing early (input is longer than necessary)
            if all(
                (self.table_symbol_to_name(table_symbol), alias_symbol) in new_tree.tables_aliases()
                for table_symbol, alias_symbol in needed_aliases
            ):
                break

        if new_tree is None:
            raise ValueError()

        # Just add any remaining tables as a left deep tree
        included_aliases = set(
            (self.table_name_to_symbol(table_name), alias_number)
            for table_name, alias_number in new_tree.tables_aliases()
        )
        for table_symbol, alias_symbol in sorted(needed_aliases - included_aliases):
            new_tree = JoinTreeBranch(
                new_tree,
                JoinTreeLeaf(self.table_symbol_to_name(table_symbol), alias_symbol),
                JoinOperator.HashJoin,
            )

        return new_tree

    def resolve_table_symbol(self, symbol_table: AliasSymbolTable, table_symbol: int) -> int:
        trees = symbol_table.get_table_aliases(table_symbol)
        if len(trees) > 0:
            return table_symbol

        query_table_symbols = list(sorted(table_symbol for table_symbol, _ in set(symbol_table.symbol_to_tree.keys())))
        return query_table_symbols[hash(table_symbol) % len(query_table_symbols)]

    def resolve_table_alias(self, symbol_table: AliasSymbolTable, table_symbol: int, alias_symbol: int) -> JoinTree:
        table_symbol = self.resolve_table_symbol(symbol_table, table_symbol)
        query_table_aliases = symbol_table.get_table_aliases(table_symbol)
        if len(query_table_aliases) == 0:
            raise ValueError("Somehow failed to resolve a valid table symbol")

        if any(table_alias == (table_symbol, alias_symbol) for table_alias in query_table_aliases):
            tree = symbol_table.get(table_symbol, alias_symbol)
            if tree is not None:
                return tree
            else:
                raise ValueError("Somehow found table alias but couldn't retrieve tree")

        # Pick one of the valid aliases
        query_table_aliases = list(sorted(query_table_aliases))
        _, resolved_alias_symbol = query_table_aliases[hash(alias_symbol) % len(query_table_aliases)]
        tree = symbol_table.get(table_symbol, resolved_alias_symbol)
        if tree is not None:
            return tree
        else:
            raise ValueError("Somehow couldn't retrieve tree for resolved alias")
