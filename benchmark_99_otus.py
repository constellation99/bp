#!/usr/bin/env python3
import argparse
import random
import time
import resource
import pandas as pd

from succinct_tree import SuccinctTree

def newick_to_bp(newick: str) -> str:
    """
    Given a Newick string, strip out everything except the parentheses
    to obtain the balanced-parentheses encoding of the tree shape.
    """
    # keep only '(' and ')'
    return ''.join(ch for ch in newick if ch in '()')

def benchmark_bp(bp: str, num_ops: int) -> dict:
    n = len(bp)

    mem0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0w  = time.perf_counter()
    t0c  = time.process_time()
    tree = SuccinctTree(bp)
    build_wall = time.perf_counter() - t0w
    build_cpu  = time.process_time()  - t0c
    mem1       = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # prepare some random queries
    idxs   = [random.randrange(n) for _ in range(num_ops)]
    ranges = [(random.randrange(n), random.randrange(n)) for _ in range(num_ops)]
    fwdqs  = [(i, 0) for i in idxs]
    bwdqs  = [(i, 0) for i in idxs]
    mincount_qs  = ranges
    minselect_qs = [(i, j, 1) for i, j in ranges]
    opens  = [i for i, b in enumerate(tree.B) if b == 1]
    # closes = [c for i in opens if (c := tree.close(i)) is not None]
    closes = []
    for i in opens:
        c = tree.close(i)
        if c is not None:
            closes.append(c)

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
    
    # warmup
    for fn, qs in tasks.items():
        if not qs:
            continue
        for _ in range(5):
            getattr(tree, fn)(*qs[0])

    # timing helper
    def measure(fn, qs):
        if not qs:
            return None
        
        # timed runs
        times = []
        for q in qs:
            st = time.perf_counter()
            getattr(tree, fn)(*q)
            times.append(time.perf_counter() - st)
        return sum(times) / len(times)

    res = {
        "n_parens":    n,
        "build_wall":  build_wall,
        "build_cpu":   build_cpu,
        "build_rss":   mem1 - mem0,
        "rmq_s":       measure("rmq",       ranges),
        "rMq_s":       measure("rMq",       ranges),
        "fwdsearch_s": measure("fwdsearch", fwdqs),
        "bwdsearch_s": measure("bwdsearch", bwdqs),
        "mincount_s":  measure("mincount",  mincount_qs),
        "minselect_s": measure("minselect", minselect_qs),
        "lca_s":       measure("lca",       [(random.choice(idxs), random.choice(idxs))
                                             for _ in range(num_ops)]),
        "close_s":     measure("close",     [(i,) for i in opens]),
        "open_s":      measure("open",      [(c,) for c in closes]),
        "enclose_s":   measure("enclose",   [(i,) for i in opens]),
    }

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    res["peak_rss_kb"] = peak_rss

    return res

def main():
    parser = argparse.ArgumentParser(description="Benchmark SuccinctTree on a Newick tree")
    parser.add_argument("newick_file", help="Path to .tree (Newick) file")
    parser.add_argument("--queries", "-q", type=int, default=5000,
                        help="Number of random queries per operation")
    args = parser.parse_args()

    # Read the Newick file
    with open(args.newick_file) as f:
        newick = f.read().strip()
    bp = newick_to_bp(newick)

    random.seed(42)
    result = benchmark_bp(bp, args.queries)

    df = pd.DataFrame([result])
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
