import argparse, os, json, random
from datasets import load_dataset, concatenate_datasets

def write_jsonl(path, rows):
    # save json files
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_conala(curated_val_size=300, use_mined=0, seed=42):
    # load curated dataset: train + test
    ds_cur = load_dataset("neulab/conala", "curated", trust_remote_code=True)
    cur_train, cur_val = ds_cur["train"], ds_cur["test"]   # curated splits
    # mined: only for 'train'
    if use_mined > 0:
        ds_m = load_dataset("neulab/conala", "mined", trust_remote_code=True)
        mined_train = ds_m["train"].shuffle(seed=seed).select(range(min(use_mined, len(ds_m["train"]))))
        train = concatenate_datasets([cur_train, mined_train])
    else:
        train = cur_train
    return train, cur_val


def main():
    # parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--val_size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_mined", type=int, default=0)   # up to 100K mined pairs: intent -> snippet
    args = ap.parse_args()

    train_ds, val_ds = load_conala(args.val_size, args.use_mined, args.seed)

    def to_row(ex):
        intent = ex.get("intent","").strip()
        snippet = ex.get("snippet","").strip()
        
        if len(intent) == 0 or len(snippet) == 0:
            return None
        else:
            return {
                "instruction": "You are an intelligent Python coding assistant. Write only valid Python code for the function, no explanations.",
                "input": intent,
                "output": snippet
            }

    train_rows = [to_row(x) for x in train_ds]
    train_rows = [x for x in train_rows if x] # remove None: when intent or snippet is None
    val_rows   = [to_row(x) for x in val_ds]
    val_rows   = [x for x in val_rows if x] # remove None

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(f"{args.out_dir}/train.jsonl", train_rows)
    write_jsonl(f"{args.out_dir}/val.jsonl",   val_rows)
    print(f"Saved {len(train_rows)} train / {len(val_rows)} val to {args.out_dir}")

if __name__ == "__main__":
    main()