"""Add pt_index column to master CSV if missing."""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if "pt_index" in df.columns:
        print("pt_index already exists, skipping.")
        return

    df.insert(0, "pt_index", range(len(df)))
    df.to_csv(args.csv, index=False)
    print(f"Added pt_index (0~{len(df)-1}) to {args.csv}")
    print(df.head(3))


if __name__ == "__main__":
    main()
