import polars as pl
from pathlib import Path

def verify_data_shift():
    data_root = Path("kaggle/input/hull-tactical-market-prediction/")
    if not data_root.exists():
        data_root = Path("/kaggle/input/hull-tactical-market-prediction/")
        
    train_path = data_root / "train_feature_selected.csv"
    if not train_path.exists():
        print("Train file not found!")
        return

    print(f"Loading {train_path}...")
    df = pl.read_csv(train_path)
    
    # 1. Show original head
    print("\n--- Original Data (First 5 rows) ---")
    print(df.select(["date_id", "forward_returns"]).head(5))

    # 2. Apply Shift Logic
    print("\n--- Applying Shift Logic ---")
    if "forward_returns" in df.columns:
        df_shifted = df.with_columns(
            pl.col("forward_returns").shift(1).alias("y")
        )
    else:
        print("Column 'forward_returns' not found!")
        return

    # 3. Show Shifted Data (Comparison)
    print("\n--- Shifted Data Comparison (First 5 rows) ---")
    comparison = df_shifted.select(["date_id", "forward_returns", "y"]).head(5)
    print(comparison)
    
    # 4. Verify Specific Logic
    # Check if y at row i equals forward_returns at row i-1
    row1_original = df["forward_returns"][0]
    row2_shifted_y = df_shifted["y"][1]
    
    print(f"\nVerification Check:")
    print(f"Original Row 0 forward_returns: {row1_original}")
    print(f"Shifted  Row 1 y              : {row2_shifted_y}")
    
    if abs(row1_original - row2_shifted_y) < 1e-9: # Float comparison
        print("SUCCESS: Shift logic is correct. y[t] == forward_returns[t-1]")
    else:
        print("FAILURE: Mismatch detected!")

    # 5. Check Data Cleaning Impact (Drop Nulls on y)
    print("\n--- Data Cleaning Check ---")
    print(f"Original Length: {len(df_shifted)}")
    df_cleaned = df_shifted.filter(pl.col("y").is_not_null())
    print(f"After dropping null y: {len(df_cleaned)}")
    print("First row of cleaned data:")
    print(df_cleaned.select(["date_id", "forward_returns", "y"]).head(1))

if __name__ == "__main__":
    verify_data_shift()

