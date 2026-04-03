import os
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import pyarrow as pa
import pyarrow.parquet as pq

print("Torch CUDA available:", torch.cuda.is_available())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()

    # Writable cache locations for Azure ML
    os.environ["HF_HOME"] = "/tmp/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
    os.environ["TORCH_HOME"] = "/tmp/torch"

    input_path = os.path.join(args.data, "data.parquet")
    output_path = os.path.join(args.out, "data.parquet")
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(input_path)

    # Basic filtering (safe + deterministic)
    df = df[df["reviewText"].notna()]
    df = df[df["reviewText"].str.len() > 0]

    model = SentenceTransformer(
        args.model_name,
        cache_folder="/tmp/huggingface",
        device="cpu"
    )

    writer = None

    with torch.no_grad():
        for start in range(0, len(df), args.chunk_size):
            end = start + args.chunk_size
            chunk = df.iloc[start:end].copy()

            texts = chunk["reviewText"].astype(str).tolist()

            embeddings = model.encode(
                texts,
                batch_size=args.batch_size,
                convert_to_numpy=True,
                show_progress_bar=(start == 0)
            )

            chunk["sbert_vector"] = embeddings.tolist()

            table = pa.Table.from_pandas(chunk, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)

            writer.write_table(table)

            # Cleanup
            del embeddings
            del chunk
            torch.cuda.empty_cache()

    if writer:
        writer.close()

    print("Rows embedded:", len(df))
    print("Embedding dimension:", model.get_sentence_embedding_dimension())


if __name__ == "__main__":
    main()
