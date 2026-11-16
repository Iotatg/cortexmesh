#!/usr/bin/env python3
"""Convert HuggingFace parquet datasets into plain text corpora for the bm tools.

  python3 tools/prep_data.py wikitext <dir> <out_train> <out_valid>
  python3 tools/prep_data.py gsm8k    <dir> <out_train> <out_valid>
  python3 tools/prep_data.py swebench <dir> <out_jsonl>
"""
import json
import sys


def write_text(rows, path, keep_newlines=False):
    n = 0
    with open(path, "w") as f:
        for row in rows:
            t = (row or "").strip()
            if not t:
                continue
            if not keep_newlines:
                t = " ".join(t.split())
            f.write(t + "\n")
            n += 1
    return n


def main():
    import pyarrow.parquet as pq

    mode, d = sys.argv[1], sys.argv[2]
    if mode == "wikitext":
        for shard, out in ((0, sys.argv[3]), (1, sys.argv[4])):
            pass
        tr = pq.read_table(f"{d}/wikitext103-train-00000-of-00002.parquet", columns=["text"])
        tr1 = pq.read_table(f"{d}/wikitext103-train-00001-of-00002.parquet", columns=["text"])
        print("train", write_text(tr.column("text").to_pylist() + tr1.column("text").to_pylist(), sys.argv[3]))
        va = pq.read_table(f"{d}/wikitext103-validation-00000-of-00001.parquet", columns=["text"])
        print("valid", write_text(va.column("text").to_pylist(), sys.argv[4]))
    elif mode == "gsm8k":
        for shard, out in ((sys.argv[3], "train"), (sys.argv[4], "valid")):
            t = pq.read_table(shard)
            q = t.column("question").to_pylist()
            a = t.column("answer").to_pylist()
            with open(out, "w") as f:
                for x, y in zip(q, a):
                    f.write("Q: " + " ".join(x.split()) + "\n")
                    f.write("A: " + " ".join(y.split()) + "\n")
                    f.write("\n")
            print(out, len(q))
    elif mode == "swebench":
        t = pq.read_table(sys.argv[3])
        cols = t.column_names
        out = sys.argv[4]
        with open(out, "w") as f:
            for i in range(t.num_rows):
                rec = {c: t.column(c)[i].as_py() for c in cols}
                if isinstance(rec.get("patch"), bytes):
                    rec["patch"] = rec["patch"].decode("utf-8", "replace")
                f.write(json.dumps(rec) + "\n")
        print("instances", t.num_rows, "cols", cols)
    else:
        raise SystemExit("unknown mode")


if __name__ == "__main__":
    main()