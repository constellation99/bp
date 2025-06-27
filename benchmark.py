#!/usr/bin/env python3
import time
import random
import argparse
import resource
import pandas as pd

from rmm_tree import BucketRMMinMax
from succinct_tree import SuccinctTree

def fmt_kb(kb: int) -> str:
    return f"{kb:,} KB"

def make_random_bp(pairs: int) -> str:
    """
    Generate a random balanced‐parentheses string with exactly `pairs` matching ().
    We do a random walk on the excess, forced never to go negative, until we use up all opens.
    """
    opens_remaining  = pairs
    closes_needed    = 0
    bp = []

    while opens_remaining or closes_needed:
        if opens_remaining > 0 and (closes_needed == 0 or random.random() < 0.6):
            bp.append('(')
            opens_remaining  -= 1
            closes_needed    += 1
        else:
            bp.append(')')
            closes_needed    -= 1

    return ''.join(bp)


def benchmark(pairs: int, num_ops: int):
    bp = make_random_bp(pairs)
    n = len(bp)
    # bits = 2 ** depth  # number of leaves; BP length ~2 * nodes
    
    # measure build
    mem0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    tree = SuccinctTree(bp)
    build_wall = time.perf_counter() - t0_wall
    build_cpu  = time.process_time() - t0_cpu
    mem1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # prepare queries
    idxs   = [random.randrange(n) for _ in range(num_ops)]
    ranges = [(random.randrange(n), random.randrange(n)) for _ in range(num_ops)]
    fwdqs  = [(i, 0) for i in idxs]
    bwdqs  = [(i, 0) for i in idxs]
    mincount_qs = ranges
    minselect_qs = [(i, j, 1) for i, j in ranges]
    opens = [i for i, b in enumerate(tree.B) if b == 1]
    closes = [c for i in opens if (c := tree.close(i)) is not None]

    results = {
        # "depth": depth,
        "pairs":      pairs,
        "n": n,
        "build_wall_s": build_wall,
        "build_cpu_s": build_cpu,
        "build_rss_kb": mem1 - mem0,
    }

    tasks = {
        "rmq":       ranges,
        "rMq":       ranges,
        "fwdsearch": fwdqs,
        "bwdsearch": bwdqs,
        "mincount":  mincount_qs,
        "minselect": minselect_qs,
        "lca":       [(random.choice(idxs), random.choice(idxs))],
        "close":     [(opens[0],)] if opens else [],
        "open":      [(closes[0],)] if closes else [],
        "enclose":   [(opens[0],)] if opens else [],
    }
    for fn, qs in tasks.items():
        if not qs:
            continue
        for _ in range(5):
            getattr(tree, fn)(*qs[0])

    def measure(fn, qs, op_name):
        if not qs:
            return None
        
        times = []
        for q in qs:
            start = time.perf_counter()
            getattr(tree, fn)(*q)
            times.append(time.perf_counter() - start)
        return sum(times) / len(times)

    results.update({
        "rmq_s":       measure("rmq",       ranges,       "rmq"),
        "rMq_s":       measure("rMq",       ranges,       "rMq"),
        "fwdsearch_s": measure("fwdsearch", fwdqs,        "fwdsearch"),
        "bwdsearch_s": measure("bwdsearch", bwdqs,        "bwdsearch"),
        "mincount_s":  measure("mincount",  mincount_qs,  "mincount"),
        "minselect_s": measure("minselect", minselect_qs, "minselect"),
        "lca_s":       measure("lca",       [(random.choice(idxs), random.choice(idxs)) for _ in range(num_ops)], "lca"),
        "close_s":     measure("close",     [(i,) for i in opens],  "close"),
        "open_s":      measure("open",      [(c,) for c in closes], "open"),
        "enclose_s":   measure("enclose",   [(i,) for i in opens],  "enclose"),
    })
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark SuccinctTree")
    parser.add_argument(
        "--pairs", "-p", type=int, nargs="+", required=True,
        help="Number of ()-pairs in the random BP to test"
    )
    parser.add_argument(
        "--queries", "-q", type=int, default=5000,
        help="Number of random queries per operation"
    )

    args = parser.parse_args()

    all_results = []
    random.seed(42)
    for pairs in args.pairs:
        # call benchmark(pairs, num_ops)
        all_results.append(benchmark(pairs, args.queries))

    df = pd.DataFrame(all_results)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()


# chmod +x benchmark.py
# /usr/bin/time -v python3 benchmark.py --pairs 100000 250000 500000 1000000 --queries 10000