import argparse
import os
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=str, required=True)
    parser.add_argument("--sentiment", type=str, required=True)
    parser.add_argument("--tfidf", type=str, required=True)
    parser.add_argument("--sbert", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    length_df = pd.read_parquet(
        args.length,
        columns=["asin", "reviewerID", "overall", "review_length_chars", "review_length_words"]
    ).set_index(["asin", "reviewerID"])

    sentiment_df = pd.read_parquet(
        args.sentiment,
        columns=[
            "asin", "reviewerID",
            "sentiment_pos", "sentiment_neg",
            "sentiment_neu", "sentiment_compound"
        ]
    ).set_index(["asin", "reviewerID"])

    tfidf_df = pd.read_parquet(
        args.tfidf,
        columns=["asin", "reviewerID", "tfidf_vector"]
    ).set_index(["asin", "reviewerID"])


    os.makedirs(args.out, exist_ok=True)
    output_path = os.path.join(args.out, "data.parquet")

    sbert_files = sorted(glob.glob(os.path.join(args.sbert, "*.parquet")))
    if not sbert_files:
        raise RuntimeError(f"No parquet files found in {args.sbert}")

    writer = None

    for parquet_file in sbert_files:
        sbert_pq = pq.ParquetFile(parquet_file)

        for i in range(sbert_pq.num_row_groups):
            table = sbert_pq.read_row_group(
                i,
                columns=["asin", "reviewerID", "sbert_vector"]
            )
            chunk = table.to_pandas()
            chunk = chunk.set_index(["asin", "reviewerID"])

            chunk = (
                chunk
                .join(length_df, how="inner")
                .join(sentiment_df, how="inner")
                .join(tfidf_df, how="inner")
            )
 # Convert to PyArrow table
            chunk.reset_index(inplace=True)

            # Sanity check
            if "overall" not in chunk.columns:
                raise RuntimeError("overall column missing after merge — check length input data")

            table = pa.Table.from_pandas(chunk)

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")

            writer.write_table(table)
            del chunk

    if writer:
        writer.close()

    print("Merge completed successfully")
    print(f"Output columns: {pq.read_schema(output_path).names}")  # add this line



if __name__ == "__main__":
    main()