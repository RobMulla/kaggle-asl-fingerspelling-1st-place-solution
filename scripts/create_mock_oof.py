import pandas as pd
import os
import shutil

# Paths
TRAIN_FOLDED = 'datamount/train_folded.csv'
SUPP_METADATA = 'datamount/supplemental_metadata_folded.csv' # Assuming this is available or we fallback
OUTPUT_FILE = 'datamount/train_folded_oof_supp.csv'

def create_mock_oof():
    print(f"Generating mock {OUTPUT_FILE}...")
    
    # 1. Load Base Train Data
    if os.path.exists(TRAIN_FOLDED):
        df = pd.read_csv(TRAIN_FOLDED)
        print(f"Loaded {TRAIN_FOLDED} with {len(df)} rows.")
    else:
        # Fallback: try to find it in kaggle input if not linked yet, or error out
        print(f"ERROR: {TRAIN_FOLDED} not found. Ensure data is linked correctly.")
        return

    # 2. Add Mock OOF Columns
    # Real script adds 'score', 'is_sup', 'phrase_len'
    # We will just fill 'score' with dummy values since we don't have weights to infer
    if 'score' not in df.columns:
        df['score'] = 0.5 # Dummy score
    
    if 'is_sup' not in df.columns:
        df['is_sup'] = 0
        
    if 'phrase_len' not in df.columns:
        df['phrase_len'] = df['phrase'].str.len()

    # 3. Add Supplemental Data (if available)
    if os.path.exists(SUPP_METADATA):
        print(f"Loading {SUPP_METADATA}...")
        supp_df = pd.read_csv(SUPP_METADATA)
        
        # Filter as per original script
        if 'phrase_len' not in supp_df.columns and 'phrase' in supp_df.columns:
             supp_df['phrase_len'] = supp_df['phrase'].str.len()
             
        # Filter short phrases
        if 'phrase_len' in supp_df.columns:
            supp_df = supp_df[supp_df['phrase_len'] < 33].copy()
            
        supp_df['score'] = 0.5
        supp_df['is_sup'] = 1
        
        # Concat
        df = pd.concat([df, supp_df], ignore_index=True)
        print(f"Added supplemental data. Total rows: {len(df)}")
    else:
        print("WARNING: Supplemental metadata not found. Skipping supplemental data addition.")

    # 4. Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    create_mock_oof()
