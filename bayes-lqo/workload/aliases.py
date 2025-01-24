from dataclasses import dataclass


@dataclass
class AliasCounts:
    """Contains the number of separate aliases for each table."""

    alias_counts: dict[str, int]

    def highest_alias(self) -> int:
        return max(self.alias_counts.values())


@dataclass
class AliasSet:
    """Contains each (table, alias number) pair."""

    aliases: set[tuple[str, int]]

    def from_alias_counts(alias_counts: AliasCounts) -> AliasSet:
        return AliasSet(
            set(
                (table, alias_num)
                for (table, num_aliases) in alias_counts.alias_counts.items()
                for alias_num in range(1, num_aliases + 1)
            )
        )
