from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
import os


hf_tknzr = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


def get_c4_data(datasets_dir, num_proc=40):
    C4_DATA_PATH = os.path.join(datasets_dir, "c4/")
    if not os.path.exists(os.path.join(C4_DATA_PATH, "train.bin")):
        os.makedirs(C4_DATA_PATH, exist_ok=True)
        dataset = load_dataset("allenai/c4", "en")

        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")

        def process(example):
            ids = hf_tknzr.encode(
                text=example["text"],
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )
            out = {"ids": ids, "len": len(ids)}
            return out

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(C4_DATA_PATH, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    return {
        "train": os.path.join(C4_DATA_PATH, "train.bin"),
        "val": os.path.join(C4_DATA_PATH, "val.bin"),
    }
