#!/usr/bin/env python3
"""Cek irisan dialogue_id antara train dan test (PsyDefConv / PsyDefDetect).

Contoh pemakaian:
    python check_overlap.py train.json test.json
    python check_overlap.py train.json test.json --key dialogue_id
    python check_overlap.py train.json test.json --key utterance_id --regex "^(.*)_\\d+$"
    python check_overlap.py train.json test.json --out overlap_ids.txt

Hanya memakai standard library Python 3.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

CANDIDATE_KEYS = [
    "dialogue_id", "dialog_id", "dialogueId", "dialogId",
    "conversation_id", "conv_id", "conversation", "dialogue", "dialog",
]
LIST_KEYS = ["data", "records", "examples", "items", "instances", "train", "test"]


def load_json(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # fallback: JSON Lines
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows


def to_records(obj):
    """Ubah isi JSON menjadi list of dict (satu dict = satu baris data)."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in LIST_KEYS:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        # dict yang key-nya adalah id dialog
        recs = []
        for k, v in obj.items():
            rec = dict(v) if isinstance(v, dict) else {"value": v}
            rec["__key__"] = k
            recs.append(rec)
        return recs
    raise ValueError("Struktur JSON tidak dikenali (bukan list atau dict).")


def get_id(rec, key, regex):
    if not isinstance(rec, dict):
        raise KeyError("Baris data bukan dict: %r" % (rec,))
    if key:
        if key not in rec:
            raise KeyError(key)
        value = rec[key]
    else:
        value = None
        for k in CANDIDATE_KEYS:
            if k in rec:
                value = rec[k]
                break
        if value is None and "__key__" in rec:
            value = rec["__key__"]
        if value is None:
            raise KeyError("<tidak ada kolom id dialog yang dikenali>")
    value = str(value)
    if regex:
        m = regex.search(value)
        if not m:
            raise ValueError("Regex tidak cocok dengan nilai id: %r" % value)
        value = m.group(1) if m.groups() else m.group(0)
    return value


def collect_ids(path, key, regex):
    records = to_records(load_json(path))
    if not records:
        sys.exit("File %s tidak berisi data." % path)
    try:
        ids = [get_id(r, key, regex) for r in records]
    except (KeyError, ValueError) as e:
        first = records[0]
        keys = list(first.keys()) if isinstance(first, dict) else type(first).__name__
        sys.exit(
            "Gagal membaca id dialog dari %s (%s).\n"
            "Kolom pada baris pertama: %s\n"
            "Gunakan --key <nama_kolom>, dan --regex kalau id dialog perlu "
            "diambil dari id ucapan." % (path, e, keys)
        )
    return ids


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train")
    ap.add_argument("test")
    ap.add_argument("--key", help="nama kolom id dialog (default: deteksi otomatis)")
    ap.add_argument("--regex",
                    help="regex untuk mengambil id dialog dari nilai kolom (group 1)")
    ap.add_argument("--out", help="simpan daftar id dialog yang beririsan ke file")
    args = ap.parse_args()

    regex = re.compile(args.regex) if args.regex else None
    train_ids = collect_ids(args.train, args.key, regex)
    test_ids = collect_ids(args.test, args.key, regex)

    train_set, test_set = set(train_ids), set(test_ids)
    shared = train_set & test_set
    train_count, test_count = Counter(train_ids), Counter(test_ids)

    test_rows_in_shared = sum(test_count[d] for d in shared)
    train_rows_in_shared = sum(train_count[d] for d in shared)

    print("=== Ringkasan ===")
    print("train: %d baris, %d dialog unik" % (len(train_ids), len(train_set)))
    print("test : %d baris, %d dialog unik" % (len(test_ids), len(test_set)))
    print()
    print("Dialog yang ada di train DAN test : %d" % len(shared))
    print("  %% dari dialog test              : %.1f%%" % pct(len(shared), len(test_set)))
    print("  %% dari dialog train             : %.1f%%" % pct(len(shared), len(train_set)))
    print("Baris test dari dialog yang juga ada di train : %d / %d (%.1f%%)"
          % (test_rows_in_shared, len(test_ids), pct(test_rows_in_shared, len(test_ids))))
    print("Baris train dari dialog yang juga ada di test : %d / %d (%.1f%%)"
          % (train_rows_in_shared, len(train_ids), pct(train_rows_in_shared, len(train_ids))))

    print()
    if not shared:
        verdict = ("Tidak ada irisan: split resmi dibuat PER DIALOG. "
                   "Skor di split resmi bersih dari dialog leakage.")
    elif len(shared) == len(test_set):
        verdict = ("Semua dialog test juga ada di train: split resmi dibuat PER UCAPAN. "
                   "Dialog yang sama muncul di train dan test.")
    else:
        verdict = ("Sebagian dialog test juga ada di train: split campuran. "
                   "Cek persentase di atas untuk melihat seberapa besar tumpang tindihnya.")
    print("Kesimpulan:", verdict)

    if shared:
        print()
        print("Contoh id dialog yang beririsan (maks. 10):")
        for d in sorted(shared)[:10]:
            print("  %s  (train: %d baris, test: %d baris)" % (d, train_count[d], test_count[d]))

    if args.out:
        Path(args.out).write_text("\n".join(sorted(shared)) + "\n", encoding="utf-8")
        print("\nDaftar id yang beririsan disimpan ke %s" % args.out)


if __name__ == "__main__":
    main()