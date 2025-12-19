import pandas as pd
import os
from tqdm import tqdm

# Paths
INPUT_CSV = 'datamount/train_folded_oof_supp.csv'
OUTPUT_CSV = 'datamount/train_folded_oof_supp_filtered.csv'
DATA_ROOT = 'datamount/train_landmarks_npy'

def filter_csv():
    print(f"Filtering {INPUT_CSV} to match files in {DATA_ROOT}...")
    
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    original_len = len(df)
    print(f"Original rows: {original_len}")
    
    # Check identifying which file_ids exist
    # We assume structure: {DATA_ROOT}/{file_id}/{sequence_id}.npy
    
    # Get unique file_ids from df
    unique_file_ids = df['file_id'].unique()
    print(f"Unique file_ids in CSV: {len(unique_file_ids)}")
    
    existing_file_ids = []
    for fid in tqdm(unique_file_ids, desc="Checking file_ids"):
        path = os.path.join(DATA_ROOT, str(fid))
        if os.path.isdir(path):
            existing_file_ids.append(fid)
            
    print(f"Found {len(existing_file_ids)} existing file_ids on disk.")
    
    if len(existing_file_ids) == 0:
        print("WARNING: No file_ids found! Check your symlinks and dataset mount.")
        # Debug listing
        print(f"Listing {DATA_ROOT} (first 5):")
        try:
            print(os.listdir(DATA_ROOT)[:5])
        except Exception as e:
            print(e)
        return

    # Filter dataframe
    df_filtered = df[df['file_id'].isin(existing_file_ids)].copy()
    print(f"Filtered rows: {len(df_filtered)} (kept {len(df_filtered)/original_len:.1%})")
    
    # Double check actual files for first few rows to be sure
    # (Optional, but good for debug)
    
    # Save
    df_filtered.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved filtered CSV to {OUTPUT_CSV}")
    
    # Update symlink or config usage
    # We can overwrite the original or ask user to update config.
    # Overwriting is risky but easiest for user "Run All" workflow if we backup.
    
    # Backup
    backup_path = INPUT_CSV + '.bak'
    if not os.path.exists(backup_path):
        os.rename(INPUT_CSV, backup_path)
        print(f"Backed up original to {backup_path}")
    
    df_filtered.to_csv(INPUT_CSV, index=False)
    print(f"Overwrote {INPUT_CSV} with filtered version.")

if __name__ == "__main__":
    filter_csv()
