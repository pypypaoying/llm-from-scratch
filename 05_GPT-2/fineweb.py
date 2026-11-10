"""Prepare a 100M-token FineWeb-Edu experiment using the course's GPT-2 tokenizer."""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset


DATA_DIR = Path(__file__).with_name("edu_fineweb100M")
TRAIN_TOKENS = 100_000_000
VAL_TOKENS = 1_000_000
SOURCE_URL = (
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/"
    "sample/10BT/000_00000.parquet"
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(documents, output_dir=DATA_DIR, train_tokens=TRAIN_TOKENS, val_tokens=VAL_TOKENS):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    iterator = iter(documents)
    manifest = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "source_url": SOURCE_URL,
        "tokenizer": "gpt2",
        "license": "odc-by",
        "split_method": "validation documents first; training starts at the next document",
        "splits": {},
    }
    started = time.monotonic()
    # The extra token supplies the final next-token target.
    for split, target_count in (("val", val_tokens + 1), ("train", train_tokens + 1)):
        temporary = output_dir / f"{split}.tmp.npy"
        array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint16, shape=(target_count,))
        position = 0
        document_count = 0
        first_id = None
        last_id = None
        next_progress = 10_000_000
        while position < target_count:
            try:
                document = next(iterator)
            except StopIteration as error:
                raise RuntimeError("The source ended before the token budget was filled") from error
            tokens = [eot] + enc.encode_ordinary(document["text"])
            assert max(tokens) < 2**16
            take = min(len(tokens), target_count - position)
            array[position:position + take] = np.asarray(tokens[:take], dtype=np.uint16)
            position += take
            document_count += 1
            document_id = document.get("id")
            first_id = document_id if document_count == 1 else first_id
            last_id = document_id
            if position >= next_progress:
                print(f"{split}: {position:,}/{target_count:,} tokens | {time.monotonic()-started:.1f}s", flush=True)
                next_progress += 10_000_000
        # Discard the unused tail of the final validation document: it must not enter training.
        array.flush()
        del array
        destination = output_dir / f"{split}.npy"
        os.replace(temporary, destination)
        manifest["splits"][split] = {
            "file": destination.name,
            "stored_tokens": target_count,
            "prediction_tokens": target_count - 1,
            "documents": document_count,
            "first_document_id": first_id,
            "last_document_id": last_id,
            "sha256": sha256(destination),
        }
        print(f"Prepared {split}: {target_count:,} stored tokens, {document_count:,} documents", flush=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    if (DATA_DIR / "manifest.json").exists():
        raise SystemExit("Prepared data already exists; keep it for a reproducible run.")
    print("Streaming the first parquet shard of FineWeb-Edu sample-10BT; preparing 1M validation + 100M training targets.", flush=True)
    documents = load_dataset(
        "parquet",
        data_files={"train": SOURCE_URL},
        columns=["text", "id"],
        split="train",
        streaming=True,
        cache_dir=str(DATA_DIR / "cache"),
    )
    prepare(documents)
