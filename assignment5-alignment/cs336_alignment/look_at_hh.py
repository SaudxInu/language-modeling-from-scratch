import pandas as pd


def main():
    filepaths = [
        "data/hh-rlhf/harmless-base/harmless-base.jsonl.gz",
        "data/hh-rlhf/helpful-online/helpful-online.jsonl.gz",
        "data/hh-rlhf/helpful-base/helpful-base.jsonl.gz",
        "data/hh-rlhf/helpful-rejection-sampled/helpful-rejection-sampled.jsonl.gz",
    ]
    dfs = []
    for filepath in filepaths:
        df = pd.read_json(filepath, lines=True, compression="gzip")
        df["source"] = filepath.split("/")[-2]
        df["num_turns"] = df["chosen"].apply(lambda x: x.count("Human: "))
        df = df[df["num_turns"] == 1]
        df["chosen"] = df["chosen"].apply(
            lambda x: "Human: " + x.split("Human: ")[1].strip()
        )
        df["rejected"] = df["rejected"].apply(
            lambda x: "Human: " + x.split("Human: ")[1].strip()
        )
        df["instruction"] = df["chosen"].apply(
            lambda x: x.split("Assistant: ")[0].strip()
        )
        df["chosen"] = df["chosen"].apply(
            lambda x: "Assistant: " + x.split("Assistant: ")[1].strip()
        )
        df["rejected"] = df["rejected"].apply(
            lambda x: "Assistant: " + x.split("Assistant: ")[1].strip()
        )
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_json(
        "data/hh-rlhf/processed_hh.jsonl.gz",
        orient="records",
        lines=True,
        compression="gzip",
    )


if __name__ == "__main__":
    main()
