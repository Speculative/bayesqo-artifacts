import os
from collections import Counter
from itertools import product, tee
from typing import Iterator

import matplotlib.pyplot as plt
import seaborn  # type: ignore
from bokeh.models import Range1d
from bokeh.palettes import Category10 as Palette
from bokeh.plotting import figure, output_file, save

from .codec import (
    Codec,
    HashProbeStackMachineCodec,
    JoinTree,
    LinearProbeStackMachineCodec,
)

TREE_TYPES = ["left_deep", "right_deep", "zigzag", "bushy"]


def tree_type(tree: JoinTree) -> str:
    if tree.is_left_deep():
        return "left_deep"
    elif tree.is_right_deep():
        return "right_deep"
    elif tree.is_zigzag():
        return "zigzag"
    else:
        return "bushy"


if __name__ == "__main__":
    # ================
    # Benchmark Config
    # ================
    NUM_TABLES = 4
    OUTPUT_DIR = "codec_benchmark"
    CODECS = [LinearProbeStackMachineCodec, HashProbeStackMachineCodec]
    # Tables that aren't in the query but are in the encoded string
    EXTRA_TABLES = 1
    # Whether to include encoded strings that are too short
    UNDER_ENCODE = True

    TOTAL_TABLES = NUM_TABLES + EXTRA_TABLES

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for codec in CODECS:
        os.makedirs(os.path.join(OUTPUT_DIR, codec.__name__), exist_ok=True)

    # "a", "b", ...
    tables = [chr(c) for c in range(ord("a"), ord("a") + TOTAL_TABLES)]

    linear_codec = LinearProbeStackMachineCodec(tables)
    hash_codec = HashProbeStackMachineCodec(tables)
    codec_instances: list[Codec] = [codec(tables) for codec in CODECS]

    codec_stats = {codec.__name__: {type: 0 for type in TREE_TYPES} for codec in CODECS}

    codec_counts: dict[str, Counter[JoinTree]] = {codec.__name__: Counter() for codec in CODECS}

    join_sizes = list(
        range(
            1 if UNDER_ENCODE else TOTAL_TABLES - 1,
            TOTAL_TABLES,
        )
    )
    print("Doing joins from size", join_sizes[0], "to", join_sizes[-1])
    for num_joins in join_sizes:
        orders: Iterator[tuple[int]] = product(range(TOTAL_TABLES), repeat=num_joins * 2)  # type: ignore
        orders, orders_copy = tee(orders)
        num_orders = sum(1 for _ in orders_copy)
        count = 0
        for order in orders:
            encoded = list(order)

            for codec_instance in codec_instances:
                decoded = codec_instance.decode(tables, encoded)
                codec_counts[codec_instance.__class__.__name__].update([decoded])
                codec_stats[codec_instance.__class__.__name__][tree_type(decoded)] += 1

            count += 1
            if count % 1_000 == 0:
                print(
                    f"{(count / num_orders) * 100:.2f}% done with {num_joins} joins",
                    end="\r",
                )

        print(end="\x1b[1K")
        print(
            f"100% done with {num_joins} joins",
        )

    # ========
    # Plotting
    # ========
    for codec in CODECS:
        rank_freq = [count for _, count in codec_counts[codec.__name__].most_common()]
        rank = list(range(len(rank_freq)))
        rank_category = [tree_type(tree) for tree, _ in codec_counts[codec.__name__].most_common()]
        category_colors = Palette[len(set(rank_category))]
        color_map = dict(zip(sorted(set(rank_category)), category_colors))
        rank_colors = [color_map[cat] for cat in rank_category]

        benchmark_info = f"{NUM_TABLES} Tables in Query, {NUM_TABLES + EXTRA_TABLES} Tables in DB{', under-encoded' if UNDER_ENCODE else ''}"

        # Create a Bokeh figure
        p = figure(
            x_range=Range1d(-1, 200, bounds=(-1, len(rank))),
            y_range=Range1d(0, max(rank_freq) * 1.1, bounds="auto"),
            title=f"{codec.__name__} Rank Frequency, \n{benchmark_info}",
            sizing_mode="stretch_both",
        )

        # Plot the bars
        data = dict(
            rank_freq=rank_freq,
            rank=rank,
            rank_category=rank_category,
            rank_colors=rank_colors,
        )
        p.vbar(
            x="rank",
            top="rank_freq",
            color="rank_colors",
            legend_field="rank_category",
            source=data,
            width=0.4,
        )

        output_file(
            os.path.join(OUTPUT_DIR, codec.__name__, "rank_freq.html"),
            title=f"{codec.__name__} Rank Frequency, \n{benchmark_info}",
        )
        save(p)

        # Visualize number of plans each type
        ax = seaborn.barplot(x=TREE_TYPES, y=[codec_stats[codec.__name__][type] for type in TREE_TYPES])
        plt.title(f"{codec.__name__} Join Types, \n{benchmark_info}")
        plt.savefig(
            os.path.join(OUTPUT_DIR, codec.__name__, "join_type.png"),
        )
        plt.clf()

        # Visualize top 10 most common trees
        os.makedirs(os.path.join(OUTPUT_DIR, codec.__name__, "top_10"), exist_ok=True)
        for rank_num, (tree, count) in enumerate(codec_counts[codec.__name__].most_common(10)):
            tree.visualize(os.path.join(OUTPUT_DIR, codec.__name__, "top_10", f"rank_{rank_num + 1}.png"))
            plt.clf()
