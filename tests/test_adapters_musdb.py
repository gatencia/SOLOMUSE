import pytest
from pathlib import Path
from solomuse_data.dataset_adapters.musdb import MusDBAdapter
from solomuse_data.config import PipelineConfig

@pytest.fixture
def fake_musdb_root(tmp_path):
    root = tmp_path / "musdb18"
    root.mkdir()
    
    # Track 1: Standard (Vocals + Band)
    t1 = root / "The Band - Song 1"
    t1.mkdir()
    (t1 / "mixture.wav").touch()
    (t1 / "vocals.wav").touch()
    (t1 / "drums.wav").touch()
    (t1 / "bass.wav").touch()
    (t1 / "other.wav").touch()
    
    # Track 2: Instrumental (No Vocals, has Other)
    t2 = root / "Instrumental Band - Jam"
    t2.mkdir()
    (t2 / "mixture.flac").touch() # Test FLAC support
    (t2 / "drums.flac").touch()
    (t2 / "bass.flac").touch()
    (t2 / "other.flac").touch()
    
    return root

@pytest.fixture
def cfg():
    return PipelineConfig(
        output_root="/tmp",
        solo_stem_policy="lead_any"
    )

def test_musdb_finds_tracks(fake_musdb_root, cfg):
    adapter = MusDBAdapter(fake_musdb_root, cfg)
    tracks = adapter.list_tracks()
    
    assert len(tracks) == 2
    ids = sorted([t.track_id for t in tracks])
    assert ids == ["Instrumental Band - Jam", "The Band - Song 1"]

def test_musdb_priority_vocals(fake_musdb_root, cfg):
    adapter = MusDBAdapter(fake_musdb_root, cfg)
    tracks = {t.track_id: t for t in adapter.list_tracks()}
    
    # Track 1 has vocals
    t1 = tracks["The Band - Song 1"]
    stems1 = adapter.list_stems(t1)
    solo1 = adapter.resolve_solo_stems(stems1)
    
    assert len(solo1) == 1
    assert solo1[0].name == "vocals.wav"

def test_musdb_priority_fallback_other(fake_musdb_root, cfg):
    adapter = MusDBAdapter(fake_musdb_root, cfg)
    tracks = {t.track_id: t for t in adapter.list_tracks()}
    
    # Track 2 has no vocals, should fallback to 'other'
    t2 = tracks["Instrumental Band - Jam"]
    stems2 = adapter.list_stems(t2)
    solo2 = adapter.resolve_solo_stems(stems2)
    
    assert len(solo2) == 1
    assert solo2[0].name == "other.flac"

def test_musdb_backing_logic(fake_musdb_root, cfg):
    adapter = MusDBAdapter(fake_musdb_root, cfg)
    tracks = {t.track_id: t for t in adapter.list_tracks()}
    
    t1 = tracks["The Band - Song 1"]
    stems1 = adapter.list_stems(t1)
    solo1 = adapter.resolve_solo_stems(stems1) # vocals
    backing1 = adapter.resolve_backing_stems(stems1, solo1)
    
    # backing should be drums, bass, other
    backing_names = sorted([p.name for p in backing1])
    assert backing_names == ["bass.wav", "drums.wav", "other.wav"]

def test_musdb_get_mix_path(fake_musdb_root, cfg):
    adapter = MusDBAdapter(fake_musdb_root, cfg)
    tracks = {t.track_id: t for t in adapter.list_tracks()}
    
    t1 = tracks["The Band - Song 1"]
    assert adapter.get_mix_path(t1).name == "mixture.wav"
    
    t2 = tracks["Instrumental Band - Jam"]
    assert adapter.get_mix_path(t2).name == "mixture.flac"
