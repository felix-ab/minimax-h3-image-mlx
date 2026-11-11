"""Pairwise relative errors between parity dumps. usage: compare.py A.npz B.npz ..."""
import sys, numpy as np, itertools
files = sys.argv[1:]
data = {f.split("/")[-1].replace(".npz", ""): np.load(f) for f in files}
keys = [k for k in ["text", "block00", "block12", "block23", "block25", "block37", "block49", "v", "h50", "layer00", "layer24", "layer48"] if all(k in d.files for d in data.values())]
for k in keys:
    print(k)
    for (a, da), (b, db) in itertools.combinations(data.items(), 2):
        x, y = da[k].astype(np.float64), db[k].astype(np.float64)
        print(f"   {a:28s} vs {b:28s} rel {np.linalg.norm(x-y)/np.linalg.norm(y):.3e}")
