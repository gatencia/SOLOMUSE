import csv
import json
from pathlib import Path
from collections import Counter
import logging
from solomuse_data.config import PipelineConfig

logger = logging.getLogger(__name__)

def inspect_intent_crashes(cfg: PipelineConfig, dataset_name: str):
    """
    Scans the intent trainer debug directories for crash JSONs and CSVs,
    tallying up the most frequent segment_ids that caused mathematical explosions.
    """
    debug_dir = Path(cfg.output_root) / "experiments" / "intent" / "debug_crashes"
    
    if not debug_dir.exists():
        print(f"✅ No crash diagnostics discovered at {debug_dir}. The trainer is stable.")
        return
        
    csv_file = debug_dir / "bad_batches.csv"
    if not csv_file.exists():
        print(f"✅ No aggregated crash CSV found at {csv_file}. ")
        return
        
    total_crashes = 0
    stage_counter = Counter()
    reason_counter = Counter()
    segment_counter = Counter()
    bad_param_counter = Counter()
    backend_counter = Counter()
    latest_timestamp = None
    
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_crashes += 1
            stage_counter[row.get("stage", "Unknown")] += 1
            reason_counter[row.get("reason", "Unknown")] += 1
            latest_timestamp = row.get("timestamp")
            
            # Count specifically what segments were involved
            segs = row.get("segment_ids", "").split("|")
            for seg in segs:
                if seg.strip():
                    segment_counter[seg.strip()] += 1
                    
    # Scan JSONs for backend and parameter level diagnostics
    json_files = list(debug_dir.glob("*.json"))
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
                
            backend = data.get("backend", "Unknown")
            backend_counter[backend] += 1
            
            grads = data.get("gradients")
            if grads and isinstance(grads, dict):
                first_bad = grads.get("first_bad_param")
                if first_bad:
                    bad_param_counter[first_bad] += 1
        except Exception as e:
            logger.warning(f"Failed to parse artifact {jf}: {e}")
                    
    print("\n" + "="*60)
    print(f"🔥 INTENT TRAINER CRASH INSPECTION REPORT")
    print("="*60)
    print(f"Total Crashes Logged: {total_crashes}")
    print(f"Latest Crash Event:   {latest_timestamp}")
    print("\nBreakdown by Stage:")
    for stage, count in stage_counter.most_common():
        print(f"  - {stage}: {count}")
        
    print("\nBreakdown by Reason:")
    for reason, count in reason_counter.most_common(5):
        print(f"  - {reason}: {count}")
        
    print("\nTop 10 Offending segment_ids:")
    for seg, count in segment_counter.most_common(10):
        print(f"  - {seg}: involved in {count} crashes")
        
    print("\nCrash Backend Distribution:")
    for backend, count in backend_counter.most_common():
        print(f"  - {backend}: {count} crashes")

    if bad_param_counter:
        print("\nTop First-Crashing Parameters:")
        for param, count in bad_param_counter.most_common(10):
            print(f"  - {param}: triggered {count} explosions")
        
    print("\nDetailed JSON artifacts available at:")
    print(f"  {debug_dir}/*.json")
    print("="*60 + "\n")
