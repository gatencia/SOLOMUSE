import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys

def verify_shapes():
    dataset = "slakh"
    print(f"--- Verifying SOLOMUSE Pipeline Tensor Shapes for {dataset} ---")
    
    # 1. Load the manifests
    manifest_base = Path(f"data/processed/segments/{dataset}/manifest.csv")
    manifest_intent = Path(f"data/processed/segments/{dataset}/manifest_intent.csv")
    manifest_renderer = Path(f"data/processed/segments/{dataset}/manifest_renderer.csv")
    
    if not manifest_base.exists() or not manifest_intent.exists() or not manifest_renderer.exists():
        print("Error: Missing manifest files. Did you run all 3 generator steps?")
        sys.exit(1)
        
    df_base = pd.read_csv(manifest_base)
    df_intent = pd.read_csv(manifest_intent)
    df_ren = pd.read_csv(manifest_renderer)
    
    # Merge them to aligned rows based on segment_id
    df = df_base.merge(df_intent, on=["dataset", "track_id", "segment_id"])
    df = df.merge(df_ren, on=["dataset", "track_id", "segment_id"])
    
    # Analyze the first 5 records
    success_count = 0
    test_limit = min(5, len(df))
    
    for i in range(test_limit):
        row = df.iloc[i]
        track_id = row['track_id']
        seg_id = row['segment_id']
        
        seg_dir = manifest_base.parent / track_id / seg_id
        sit_file = seg_dir / "situation.npy"
        int_file = seg_dir / "intent_targets.npy"
        ren_file = seg_dir / "renderer_target.npy"
        
        print(f"\n[{seg_id}] Validating arrays...")
        
        if not sit_file.exists() or not int_file.exists() or not ren_file.exists():
            print(f"  ❌ Missing NumPy file(s) in {seg_dir}")
            continue
            
        sit_arr = np.load(sit_file)
        int_arr = np.load(int_file)
        ren_arr = np.load(ren_file)
        
        # Log the True Shapes
        print(f"  Load Success! Actual Memory Shapes:")
        print(f"    - Situation: {sit_arr.shape}")
        print(f"    - Intent:    {int_arr.shape}")
        print(f"    - Renderer:  {ren_arr.shape}")
        
        # Compare to manifest declarations
        expected_intent_frames = row['intent_frames']
        expected_ren_frames = row['renderer_frames']
        expected_ren_dim = row['renderer_dim']
        
        errs = []
        if int_arr.shape[0] != expected_intent_frames:
            errs.append(f"Intent frames mismatch: manifest={expected_intent_frames}, array={int_arr.shape[0]}")
        if ren_arr.shape[0] != expected_ren_frames:
            errs.append(f"Renderer frames mismatch: manifest={expected_ren_frames}, array={ren_arr.shape[0]}")
        if ren_arr.shape[1] != expected_ren_dim:
            errs.append(f"Renderer dim mismatch: manifest={expected_ren_dim}, array={ren_arr.shape[1]}")
            
        if errs:
            print(f"  ❌ CSV-to-Array Alignment Failed:")
            for e in errs:
                print(f"     {e}")
        else:
            print(f"  ✅ Perfect Alignment. CSV manifest exactly matches disk arrays.")
            success_count += 1
            
    print(f"\n--- Verification Complete: {success_count}/{test_limit} passed ---")

if __name__ == "__main__":
    verify_shapes()
