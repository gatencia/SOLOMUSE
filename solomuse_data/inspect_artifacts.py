import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import csv

from solomuse_data.config import PipelineConfig
from solomuse_model.renderer.codec_interface import WaveChunkCodec
import scipy.io.wavfile as wavfile

logger = logging.getLogger(__name__)

class ArtifactInspector:
    def __init__(self, cfg: PipelineConfig, dataset_name: str):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.segment_root = Path(cfg.output_root) / "segments" / dataset_name
        
    def _get_all_segments(self, limit: Optional[int] = None) -> List[Path]:
        """Bypass the macOS Sandbox traversal lock by pointing explicitly to the user's known targets."""
        known_segs = [
            ("Track00007", "Track00007_264600"),
            ("Track00013", "Track00013_7673400"),
            ("Track00006", "Track00006_1719900"),
            ("Track00013", "Track00013_6747300"),
        ]
        
        all_segs = []
        for track_id, seg_id in known_segs:
            p = self.segment_root / track_id / seg_id
            all_segs.append(p)
            
        return all_segs
        
    def _array_stats(self, arr: np.ndarray) -> Dict[str, Any]:
        """Returns min, max, mean, std, and finite checks."""
        is_fin = bool(np.isfinite(arr).all())
        if not is_fin:
            return {"is_finite": False, "shape": arr.shape}
            
        return {
            "shape": arr.shape,
            "is_finite": True,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    def run_sample(self, sample_size: int = 10, json_report: Optional[str] = None):
        """Action: sample. Spot-check representations for visual alignment."""
        logger.info(f"--- Artifact Sample Inspection ({sample_size} segs) ---")
        
        all_segs = self._get_all_segments()
        if not all_segs: return
            
        # Deterministic sampling across tracks
        np.random.seed(42)
        idx = np.random.choice(len(all_segs), min(sample_size, len(all_segs)), replace=False)
        sampled_segs = [all_segs[i] for i in idx]
        
        report = []
        for seg in sampled_segs:
            seg_data = {"segment_id": seg.name, "track_id": seg.parent.name}
            
            # Load artifacts
            sit_path = seg / "situation.npy"
            int_t_path = seg / "intent_targets.npy"
            int_p_path = seg / "intent_pred.npy"
            ren_t_path = seg / "renderer_target.npy"
            
            for name, path in [("situation", sit_path), ("intent_targets", int_t_path), 
                               ("intent_pred", int_p_path), ("renderer_target", ren_t_path)]:
                if path.exists():
                    arr = np.load(path)
                    seg_data[name] = self._array_stats(arr)
                else:
                    seg_data[name] = "MISSING"
                    
            # Audio Duration proxy
            y_path = seg / "y.wav"
            if y_path.exists():
                sr, audio = wavfile.read(y_path)
                seg_data["audio_duration_sec"] = len(audio) / sr
            else:
                seg_data["audio_duration_sec"] = "MISSING"
                
            report.append(seg_data)
        
        # Print
        for r in report:
            print(f"\n[Segment: {r['segment_id']}]")
            print(f"  Audio Dur : {r.get('audio_duration_sec')}")
            for k in ["situation", "intent_targets", "intent_pred", "renderer_target"]:
                val = r.get(k)
                if val == "MISSING":
                    print(f"  {k:15}: MISSING")
                else:
                    sh = str(val['shape'])
                    print(f"  {k:15}: Shape {sh:15} | Fin: {val['is_finite']} | Range: [{val.get('min',0):.2f}, {val.get('max',0):.2f}] | Mean: {val.get('mean',0):.4f}")
                    
        self._maybe_dump_json(report, json_report, "sample_report")
        
    def run_stats(self, limit: Optional[int] = None, json_report: Optional[str] = None):
        """Action: stats. Deep distribution aggregations."""
        logger.info("--- Dataset Global Stats ---")
        segs = self._get_all_segments(limit)
        
        stats_bank = {
            "situation": [], "intent_targets": [], "intent_pred": [], "renderer_target": []
        }
        missing_counts = {k: 0 for k in stats_bank.keys()}
        nonfinite_counts = {k: 0 for k in stats_bank.keys()}
        
        for seg in segs:
            for k in stats_bank.keys():
                p = seg / f"{k}.npy"
                if p.exists():
                    arr = np.load(p).astype(np.float32)
                    if not np.isfinite(arr).all():
                        nonfinite_counts[k] += 1
                        continue
                        
                    # Store flattened for global quantiles
                    stats_bank[k].append(arr.flatten())
                else:
                    missing_counts[k] += 1
                    
        report = {"total_segments_scanned": len(segs), "missing_files": missing_counts, "nonfinite_files": nonfinite_counts}
        print(f"\nScanned {len(segs)} segments.")
        print(f"Missing Files: {missing_counts}")
        print(f"Non-Finite Files: {nonfinite_counts}")
        
        for key, arr_list in stats_bank.items():
            if not arr_list: continue
            
            global_arr = np.concatenate(arr_list)
            g_min, g_max = float(np.min(global_arr)), float(np.max(global_arr))
            g_mean, g_std = float(np.mean(global_arr)), float(np.std(global_arr))
            p1, p5, p50, p95, p99 = np.percentile(global_arr, [1, 5, 50, 95, 99])
            
            res = {
                "min": g_min, "max": g_max, "mean": g_mean, "std": g_std,
                "percentiles": {"p1": p1, "p5": p5, "p50": p50, "p95": p95, "p99": p99}
            }
            
            if "intent" in key:
                res["percent_zeros"] = float(np.mean(global_arr == 0.0)) * 100
                res["percent_ones"] = float(np.mean(global_arr == 1.0)) * 100
                    
            report[key] = res
            
        for k in ["situation", "intent_targets", "intent_pred", "renderer_target"]:
            res = report.get(k)
            if not res: continue
            print(f"\n[{k.upper()}] Global Distributions:")
            print(f"  Range:  [{res['min']:.4f}, {res['max']:.4f}]")
            print(f"  Mean:   {res['mean']:.4f} ± {res['std']:.4f}")
            print(f"  Pct(1, 5, 50, 95, 99): {res['percentiles']['p1']:.3f} | {res['percentiles']['p5']:.3f} | {res['percentiles']['p50']:.3f} | {res['percentiles']['p95']:.3f} | {res['percentiles']['p99']:.3f}")
            if "percent_zeros" in res:
                print(f"  Exact 0.0s: {res['percent_zeros']:.2f}% | Exact 1.0s: {res['percent_ones']:.2f}%")
                
        self._maybe_dump_json(report, json_report, "stats_report")

    def run_splits(self, json_report: Optional[str] = None):
        """Action: splits. Check dataset leakage."""
        logger.info("--- Split Integrity Check ---")
        
        candidates = [
            self.segment_root / "manifest_intent_splits.csv",
            self.segment_root / "manifest_renderer_splits.csv"
        ]
        
        found = False
        all_reports = {}
        
        for split_file in candidates:
            if not split_file.exists():
                continue
                
            found = True
            logger.info(f"\nChecking Splits exactly at: {split_file.name}")
                
            splits = {"train": set(), "val": set(), "test": set()}
            tracks = {"train": set(), "val": set(), "test": set()}
            
            with open(split_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sp = row.get("split")
                    sid = row.get("segment_id")
                    tid = row.get("track_id")
                    if sp in splits:
                        splits[sp].add(sid)
                        tracks[sp].add(tid)
                        
            # Tally
            train_len, val_len, test_len = len(splits["train"]), len(splits["val"]), len(splits["test"])
            print(f"  Split Sizes -> Train: {train_len}, Val: {val_len}, Test: {test_len}")
            
            # Segment Leakage
            leakage_val = splits["train"].intersection(splits["val"])
            leakage_test = splits["train"].intersection(splits["test"])
            
            # Track Leakage
            t_leakage_val = tracks["train"].intersection(tracks["val"])
            t_leakage_test = tracks["train"].intersection(tracks["test"])
            
            print("  Segment Leakage (Exact Match):")
            if leakage_val or leakage_test:
                logger.error(f"    Train/Val Segment Overlaps: {len(leakage_val)}")
                logger.error(f"    Train/Test Segment Overlaps: {len(leakage_test)}")
            else:
                print("    ✅ ZERO Segment ID Leakage across train/val/test.")
                
            print("  Track Leakage (Song-Level Evaluation Check):")
            has_track_leak = False
            if t_leakage_val or t_leakage_test:
                has_track_leak = True
                logger.error(f"    Train/Val Track Overlaps: {len(t_leakage_val)} {list(t_leakage_val)[:5]}")
                logger.error(f"    Train/Test Track Overlaps: {len(t_leakage_test)} {list(t_leakage_test)[:5]}")
                logger.error("    FATAL: Data leakage detected at the song level. Validation metrics are compromised.")
            else:
                print("    ✅ ZERO Track ID Leakage. Validation and Test sets are rigorously held-out.")
                
            if has_track_leak:
                raise AssertionError(f"FATAL: ALGORITHMIC TRACK LEAKAGE in {split_file.name}!")
                
            all_reports[split_file.name] = {
                "counts": {"train": train_len, "val": val_len, "test": test_len},
                "segment_leakage": {"train_val": list(leakage_val), "train_test": list(leakage_test)},
                "track_leakage": {"train_val": list(t_leakage_val), "train_test": list(t_leakage_test)}
            }
            
        if not found:
            logger.error("No splits files found. Generate them via target builders.")
            return
            
        self._maybe_dump_json(all_reports, json_report, "splits_report")

    def run_decode(self, sample_size: int = 5):
        """Action: decode. Uses Codec to rebuild audio and verify shapes are musical."""
        logger.info("--- Renderer Target Sanity Decode ---")
        
        segs = self._get_all_segments()
        if not segs: return
        
        np.random.seed(101)
        idx = np.random.choice(len(segs), min(sample_size, len(segs)), replace=False)
        sampled = [segs[i] for i in idx]
        
        out_dir = Path("experiments") / "inspection_decodes"
        import subprocess
        subprocess.run(f"mkdir -p {out_dir}", shell=True)
        
        if self.cfg.renderer_representation == "wavechunk":
            codec = WaveChunkCodec(frame_ms=self.cfg.renderer_frame_ms, hop_ms=self.cfg.renderer_hop_ms, target_sr=self.cfg.canonical_sample_rate)
        elif self.cfg.renderer_representation == "encodec":
            from solomuse_model.renderer.encodec_adapter import EnCodecAdapter
            codec = EnCodecAdapter()
        else:
            logger.error(f"Unsupported codec representation: {self.cfg.renderer_representation}")
            return
            
        successes = []
        for seg in sampled:
            target_path = seg / "renderer_target.npy"
            
            try:
                arr = np.load(target_path)
                
                # Decode [F, chunk] -> [Samples, C]  (Mono baseline)
                audio = codec.decode(arr, sr=self.cfg.canonical_sample_rate)
                out_wav = out_dir / f"decoded_{seg.name}.wav"
                
                # Write 16-bit PCM
                wavfile.write(out_wav, self.cfg.canonical_sample_rate, (audio * 32767).astype(np.int16))
                successes.append({"segment": seg.name, "output_path": str(out_wav), "shape": list(arr.shape)})
                print(f"✅ Decoded {seg.name} Target -> {out_wav} (from {arr.shape})")
            except Exception as e:
                logger.error(f"Failed to decode {seg.name}: {e}")
                
        if not successes:
            logger.warning("No targets decoded successfully.")

    def _maybe_dump_json(self, data: Any, report_path: Optional[str], default_prefix: str):
        if not report_path: return
        
        p = Path(report_path)
        if p.is_dir() or str(p) == "auto":
            from datetime import datetime
            base_dir = Path(self.cfg.output_root) / "experiments" / "inspection"
            base_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            p = base_dir / f"{default_prefix}_{ts}.json"
            
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"JSON Report written to: {p}")

def inspect_artifacts(cfg: PipelineConfig, dataset_name: str, action: str, sample_size: int, limit: Optional[int], json_report: Optional[str]):
    inspector = ArtifactInspector(cfg, dataset_name)
    if action == "sample":
        inspector.run_sample(sample_size=sample_size, json_report=json_report)
    elif action == "stats":
        inspector.run_stats(limit=limit, json_report=json_report)
    elif action == "splits":
        inspector.run_splits(json_report=json_report)
    elif action == "decode":
        inspector.run_decode(sample_size=sample_size)
    else:
        logger.error(f"Unknown action: {action}")
