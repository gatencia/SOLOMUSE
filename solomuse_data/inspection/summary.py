
import os
import json
import csv
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import Counter

logger = logging.getLogger(__name__)

class UnifiedSummaryExporter:
    """
    Computes diagnostic aggregations from a UnifiedArtifactExporter report.
    Supports filtering by 'generated' and 'clean' subsets.
    """
    def __init__(self, output_root: str):
        self.output_root = Path(output_root)
        self.summary_out_dir = self.output_root / "experiments" / "inspection_summary"
        
    def load_data(self, report_path: str) -> List[Dict[str, Any]]:
        """Loads data from CSV or JSON report."""
        path = Path(report_path)
        if not path.exists():
            raise FileNotFoundError(f"Report file not found: {report_path}")
            
        if path.suffix == ".json":
            with open(path, "r") as f:
                return json.load(f)
        elif path.suffix == ".csv":
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                data = list(reader)
                # Convert numeric strings back to types
                for row in data:
                    for k, v in row.items():
                        if v == "": continue
                        if k.startswith("has_") or k in ["alignment_ok", "y_is_silent", "renderer_target_all_zero"]:
                            row[k] = int(v) if v.isdigit() else v
                        elif any(s in k for s in ["_rms", "_peak", "_mse", "_sim", "_snr", "_mean", "_std", "_min", "_max"]):
                            try: row[k] = float(v)
                            except: pass
                return data
        else:
            raise ValueError(f"Unsupported report format: {path.suffix}")

    def compute_subset_metrics(self, data: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
        """Compute metrics for a specific subset of the data."""
        total = len(data)
        if total == 0:
            return {"subset": label, "count": 0}

        counts = {
            "split_unknown": 0,
            "has_situation": 0,
            "has_intent_targets": 0,
            "has_renderer_target": 0,
            "renderer_target_all_zero": 0,
            "y_is_silent": 0,
            "suspicious_zero_renderer": 0, # zero renderer but NOT silent
            "clean_rows": 0
        }
        
        splits = Counter()
        
        for r in data:
            sp = r.get("split", "UNKNOWN")
            splits[sp] += 1
            if sp == "UNKNOWN": counts["split_unknown"] += 1
            
            for k in ["has_situation", "has_intent_targets", "has_renderer_target", 
                      "renderer_target_all_zero", "y_is_silent"]:
                if r.get(k): counts[k] += 1
                
            if r.get("renderer_target_all_zero") and not r.get("y_is_silent"):
                counts["suspicious_zero_renderer"] += 1
                
            # rows_clean := has_situation=1 AND has_intent_targets=1 AND has_renderer_target=1 AND alignment_ok=1
            is_clean = (r.get("has_situation") == 1 and 
                        r.get("has_intent_targets") == 1 and 
                        r.get("has_renderer_target") == 1 and 
                        r.get("alignment_ok") == 1)
            if is_clean:
                counts["clean_rows"] += 1

        metrics = {
            "subset": label,
            "total_count": total,
            "counts": counts,
            "split_distribution": dict(splits)
        }
        
        # Distribution stats for clean rows
        clean_rows = [r for r in data if (r.get("has_situation") == 1 and 
                                          r.get("has_intent_targets") == 1 and 
                                          r.get("has_renderer_target") == 1 and 
                                          r.get("alignment_ok") == 1)]
        
        dist_stats = {}
        cols_to_stat = ["situation_mean", "intent_targets_mean", "renderer_target_std"]
        for col in cols_to_stat:
            vals = [r[col] for r in clean_rows if col in r and isinstance(r[col], (int, float))]
            if vals:
                v_arr = np.array(vals)
                dist_stats[col] = {
                    "min": float(np.min(v_arr)),
                    "max": float(np.max(v_arr)),
                    "mean": float(np.mean(v_arr)),
                    "std": float(np.std(v_arr)),
                    "p1": float(np.percentile(v_arr, 1)),
                    "p50": float(np.percentile(v_arr, 50)),
                    "p99": float(np.percentile(v_arr, 99))
                }
        
        metrics["distribution_stats"] = dist_stats
        return metrics

    def validate_split_integrity(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect track-level split leakage and segment duplicates."""
        track_to_splits = {}
        seg_counts = Counter()
        
        for r in data:
            sid = r.get("segment_id")
            tid = r.get("track_id")
            sp = r.get("split", "UNKNOWN")
            
            if sp != "UNKNOWN":
                if tid not in track_to_splits: track_to_splits[tid] = set()
                track_to_splits[tid].add(sp)
            
            if sid: seg_counts[sid] += 1
            
        leaking_tracks = {tid: list(s) for tid, s in track_to_splits.items() if len(s) > 1}
        duplicate_segments = {sid: count for sid, count in seg_counts.items() if count > 1}
        
        return {
            "leaking_tracks_count": len(leaking_tracks),
            "top_leaking_tracks": dict(list(leaking_tracks.items())[:20]),
            "duplicate_segments_count": len(duplicate_segments),
            "top_duplicate_segments": dict(list(duplicate_segments.items())[:20])
        }

    def run_summary(self, report_path: str):
        """Processes report into summary outputs."""
        data = self.load_data(report_path)
        
        # 1. Subsets
        all_metrics = self.compute_subset_metrics(data, "all")
        
        generated_data = [r for r in data if r.get("has_situation") or r.get("has_intent_targets") or r.get("has_renderer_target")]
        generated_metrics = self.compute_subset_metrics(generated_data, "only-generated")
        
        clean_data = [r for r in data if (r.get("has_situation") == 1 and 
                                         r.get("has_intent_targets") == 1 and 
                                         r.get("has_renderer_target") == 1 and 
                                         r.get("alignment_ok") == 1)]
        clean_metrics = self.compute_subset_metrics(clean_data, "only-clean")
        
        # 2. Integrity
        integrity = self.validate_split_integrity(data)
        
        # 3. Split Source Audit
        # We assume the dataset name is needed to find the segments folder
        # But we can try to guess it from the report_path or just check the output_root
        # Better: let's look for any manifest_*_splits.csv in the output_root/segments/*/
        dataset_guess = Path(report_path).stem.split('_')[-1] # very rough
        # Actually, let's just check if ANY split files exist in common locations
        split_source = "missing"
        # Search segment_root/dataset/manifest_intent_splits.csv
        # Since we don't have the dataset name for sure, let's just check if the data has splits
        if any(r.get("split") and r.get("split") != "UNKNOWN" for r in data):
            split_source = "found"
        
        # If we want to be more specific, we'd need the dataset name.
        # But the prompt says: "If a split file is missing, the code should clearly say so in summary.json (“split_source”: “missing”)."
        # If we are loading an existing report, and it has NO splits (all UNKNOWN), it's missing.
        
        summary_data = {
            "report_source": report_path,
            "split_source": split_source,
            "subsets": {
                "all": all_metrics,
                "only-generated": generated_metrics,
                "only-clean": clean_metrics
            },
            "integrity": integrity
        }
        
        self.write_summary(summary_data)
        self.print_summary(summary_data)
        
    def write_summary(self, summary_data: Dict[str, Any]):
        """Writes JSON and CSV summary files."""
        self.summary_out_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = self.summary_out_dir / "summary.json"
        with open(json_path, "w") as f:
            json.dump(summary_data, f, indent=2)
            
        csv_path = self.summary_out_dir / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Header with some key metrics
            writer.writerow(["Subset", "Total", "Clean", "SplitUNKNOWN", "ZeroRenderer", "SilentY", "SuspiciousZero"])
            for label, metrics in summary_data["subsets"].items():
                c = metrics["counts"]
                writer.writerow([
                    label, metrics["total_count"], c["clean_rows"], c["split_unknown"],
                    c["renderer_target_all_zero"], c["y_is_silent"], c["suspicious_zero_renderer"]
                ])
                
        logger.info(f"Summary written to: {self.summary_out_dir}")

    def print_summary(self, summary_data: Dict[str, Any]):
        """Prints a concise summary to console."""
        print("\n" + "="*70)
        print("INSPECTION SUMMARY REPORT")
        print("="*70)
        
        for label, metrics in summary_data["subsets"].items():
            c = metrics["counts"]
            print(f"[{label.upper():<15}] Total: {metrics['total_count']:<6} | Clean: {c['clean_rows']:<6} | UNKNOWN Split: {c['split_unknown']}")
            if label == "all" and metrics["total_count"] > 0:
                 print(f"  Artifact Coverage: Situation={c['has_situation']} | Intent={c['has_intent_targets']} | Renderer={c['has_renderer_target']}")
                 print(f"  Pathologies      : ZeroRenderer={c['renderer_target_all_zero']} | SilentY={c['y_is_silent']} | Suspicious={c['suspicious_zero_renderer']}")

        integrity = summary_data["integrity"]
        print("\nINTEGRITY AUDIT:")
        print(f"  Split Source      : {summary_data['split_source']}")
        print(f"  Duplicate Segments: {integrity['duplicate_segments_count']}")
        if integrity["leaking_tracks_count"] > 0:
            print(f"  [WARNING] Track Leakage: {integrity['leaking_tracks_count']} tracks in multiple splits!")
        else:
            print(f"  [OK] Track Leakage: 0 tracks (strictly disjoint splits).")
            
        print("\nOUTPUT PATHS:")
        print(f"  JSON: {self.summary_out_dir}/summary.json")
        print(f"  CSV : {self.summary_out_dir}/summary.csv")
        print("="*70 + "\n")
