
import json
import csv
import numpy as np
import pytest
import tempfile
from pathlib import Path
from solomuse_data.inspection.summary import UnifiedSummaryExporter

@pytest.fixture
def mock_report_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        report_data = [
            # 1. Clean row in train
            {
                "segment_id": "seg1", "track_id": "trackA", "split": "train",
                "has_situation": 1, "has_intent_targets": 1, "has_renderer_target": 1, "alignment_ok": 1,
                "renderer_target_all_zero": 0, "y_is_silent": 0,
                "situation_mean": 0.5, "intent_targets_mean": 0.1, "renderer_target_std": 2.0
            },
            # 2. Row with split unknown
            {
                "segment_id": "seg2", "track_id": "trackB", "split": "UNKNOWN",
                "has_situation": 1, "has_intent_targets": 0, "has_renderer_target": 0, "alignment_ok": 1,
                "renderer_target_all_zero": 0, "y_is_silent": 0
            },
            # 3. Row all-zero renderer target but NOT silent (suspicious)
            {
                "segment_id": "seg3", "track_id": "trackC", "split": "test",
                "has_situation": 1, "has_intent_targets": 1, "has_renderer_target": 1, "alignment_ok": 1,
                "renderer_target_all_zero": 1, "y_is_silent": 0,
                "situation_mean": -0.1, "intent_targets_mean": 0.5, "renderer_target_std": 0.0
            },
            # 4. Silent row
            {
                "segment_id": "seg4", "track_id": "trackD", "split": "val",
                "has_situation": 1, "has_intent_targets": 1, "has_renderer_target": 1, "alignment_ok": 1,
                "renderer_target_all_zero": 1, "y_is_silent": 1,
                "situation_mean": 0.0, "intent_targets_mean": 0.0, "renderer_target_std": 0.0
            },
            # 5. Not-generated row (no artifacts)
            {
                "segment_id": "seg5", "track_id": "trackE", "split": "train",
                "has_situation": 0, "has_intent_targets": 0, "has_renderer_target": 0, "alignment_ok": 0
            },
            # 6. Leaking track (trackA in both train and val)
            {
                "segment_id": "seg6", "track_id": "trackA", "split": "val",
                "has_situation": 1, "has_intent_targets": 1, "has_renderer_target": 1, "alignment_ok": 1
            }
        ]
        
        json_path = tmp_path / "report.json"
        with open(json_path, "w") as f:
            json.dump(report_data, f)
            
        yield json_path, tmp_path

def test_summary_counts_consistent_with_report(mock_report_path):
    report_path, tmp_path = mock_report_path
    exporter = UnifiedSummaryExporter(str(tmp_path))
    exporter.run_summary(str(report_path))
    
    summary_json = tmp_path / "experiments" / "inspection_summary" / "summary.json"
    assert summary_json.exists()
    
    with open(summary_json, "r") as f:
        res = json.load(f)
        
    all_metrics = res["subsets"]["all"]
    counts = all_metrics["counts"]
    
    assert all_metrics["total_count"] == 6
    assert counts["split_unknown"] == 1
    assert res["split_source"] == "found"
    assert counts["clean_rows"] == 4
    
    assert counts["suspicious_zero_renderer"] == 1 # seg3
    assert counts["y_is_silent"] == 1 # seg4 (seg3 is zero renderer but NOT silent)

def test_only_generated_filter(mock_report_path):
    report_path, tmp_path = mock_report_path
    exporter = UnifiedSummaryExporter(str(tmp_path))
    exporter.run_summary(str(report_path))
    
    with open(tmp_path / "experiments" / "inspection_summary" / "summary.json", "r") as f:
        res = json.load(f)
        
    gen_metrics = res["subsets"]["only-generated"]
    # Only seg5 has no artifacts. So 6-1 = 5.
    assert gen_metrics["total_count"] == 5

def test_only_clean_filter(mock_report_path):
    report_path, tmp_path = mock_report_path
    exporter = UnifiedSummaryExporter(str(tmp_path))
    exporter.run_summary(str(report_path))
    
    with open(tmp_path / "experiments" / "inspection_summary" / "summary.json", "r") as f:
        res = json.load(f)
        
    clean_metrics = res["subsets"]["only-clean"]
    assert clean_metrics["total_count"] == 4

def test_split_integrity_validation(mock_report_path):
    report_path, tmp_path = mock_report_path
    exporter = UnifiedSummaryExporter(str(tmp_path))
    exporter.run_summary(str(report_path))
    
    with open(tmp_path / "experiments" / "inspection_summary" / "summary.json", "r") as f:
        res = json.load(f)
        
    integrity = res["integrity"]
    assert integrity["leaking_tracks_count"] == 1
    assert "trackA" in integrity["top_leaking_tracks"]

def test_distribution_stats(mock_report_path):
    report_path, tmp_path = mock_report_path
    exporter = UnifiedSummaryExporter(str(tmp_path))
    exporter.run_summary(str(report_path))
    
    with open(tmp_path / "experiments" / "inspection_summary" / "summary.json", "r") as f:
        res = json.load(f)
        
    dist = res["subsets"]["all"]["distribution_stats"]
    assert "situation_mean" in dist
    assert dist["situation_mean"]["min"] == -0.1
    assert dist["situation_mean"]["max"] == 0.5
