import polars as pl
from pathlib import Path

def analyze_data_quality():
    data_root = Path("kaggle/input/hull-tactical-market-prediction/")
    if not data_root.exists():
        data_root = Path("/kaggle/input/hull-tactical-market-prediction/")
        
    train_path = data_root / "train_feature_selected.csv"
    
    if not train_path.exists():
        print("Data file not found!")
        return

    print(f"Loading {train_path}...")
    df = pl.read_csv(train_path)
    
    # Update features based on user provided CSV header
    # "date_id,E2,M13,P8,P5,V9,S2,M12,S5,forward_returns"
    feature_cols = ["E2", "M13", "P8", "P5", "V9", "S2", "M12", "S5"]
    total_rows = len(df)
    print(f"Total rows: {total_rows}")
    
    # 1. Count nulls per row for feature columns
    null_counts = df.select([
        pl.sum_horizontal([pl.col(c).is_null() for c in feature_cols]).alias("null_count")
    ])
    
    # Add null_count column to df
    df_analyzed = df.with_columns(null_counts)
    
    print("\n--- Null Count Distribution ---")
    dist = df_analyzed.group_by("null_count").len().sort("null_count")
    print(dist)
    
    # 2. Simulate Thresholds
    thresholds = [0, 2, 4, 6, 8] # Max nulls allowed
    
    print("\n--- Retention Simulation ---")
    for t in thresholds:
        # Keep rows where null_count <= t
        kept_count = df_analyzed.filter(pl.col("null_count") <= t).height
        dropped = total_rows - kept_count
        print(f"Max Nulls Allowed = {t}: Keep {kept_count} rows ({kept_count/total_rows:.1%}), Drop {dropped}")

    # 3. Inspect the structure of 'missingness'
    print("\n--- Row Range Analysis ---")
    high_null_rows = df_analyzed.with_row_index().filter(pl.col("null_count") > 4)
    if len(high_null_rows) > 0:
        max_idx = high_null_rows.select(pl.col("index").max()).item()
        min_idx = high_null_rows.select(pl.col("index").min()).item()
        count = len(high_null_rows)
        print(f"Rows with >4 nulls: {count} rows.")
        print(f"Index range of bad rows: {min_idx} to {max_idx}")
    else:
        print("No rows with >4 nulls found.")

if __name__ == "__main__":
    analyze_data_quality()

