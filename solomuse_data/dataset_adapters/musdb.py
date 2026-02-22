import logging
from pathlib import Path
from typing import List, Dict, Optional
from solomuse_data.dataset_adapters.base import DatasetAdapter, Track
from solomuse_data.config import PipelineConfig

logger = logging.getLogger(__name__)

class MusDBAdapter(DatasetAdapter):
    """
    Adapter for MUSDB18-like datasets.
    Typical structure:
      train/
        A Classic Education - NightOwl/
           mixture.wav
           vocals.wav
           drums.wav
           bass.wav
           other.wav
    """
    
    def __init__(self, root: Path, cfg: PipelineConfig):
        super().__init__(root, cfg)
        
    def list_tracks(self) -> List[Track]:
        """
        Find directory based tracks. MUSDB tracks are folders containing 'mixture.wav' (or similar).
        """
        tracks = []
        if not self.root.exists():
            logger.warning(f"MUSDB root {self.root} does not exist.")
            return []
            
        # Recursive walk
        for path in self.root.rglob("*"):
             if path.is_dir():
                 # Check for mixture file
                 has_mix = any((path / f).exists() for f in ["mixture.wav", "mix.wav", "mixture.flac", "mix.flac"])
                 # Check for stems?
                 has_stems = (path / "vocals.wav").exists() or (path / "vocals.stem.wav").exists() # etc
                 
                 if has_mix or has_stems:
                     # Use relative path from root as ID? Or just folder name? 
                     # MUSDB folder names are unique usually.
                     tracks.append(Track(dataset="musdb", track_id=path.name, root=path))
                     
        return tracks

    def list_stems(self, track: Track) -> Dict[str, Path]:
        stems = {}
        # Standard MUSDB stems
        # Try wav and flac
        candidates = ["vocals", "drums", "bass", "other", "accompaniment"]
        extensions = [".wav", ".flac"]
        
        for name in candidates:
            for ext in extensions:
                p = track.root / (name + ext)
                if p.exists():
                    stems[name] = p
                    break # Found this stem, stop checking extensions
        
        return stems

    def get_mix_path(self, track: Track) -> Optional[Path]:
        for name in ["mixture.wav", "mix.wav", "mixture.flac", "mix.flac"]:
            p = track.root / name
            if p.exists():
                return p
        return None

    def get_metadata(self, track: Track) -> Dict:
        meta = super().get_metadata(track)
        meta["license"] = "varies (research)" # MUSDB is research-only usually
        return meta

    def resolve_solo_stems(self, stems: Dict[str, Path]) -> List[Path]:
        """
        Override for MUSDB specific logic.
        SOLO=lead_any:
          1. Vocals (highest priority)
          2. Other (fallback if no vocals)
        """
        if self.cfg.solo_stem_policy != "lead_any":
            return []

        # Check for vocals
        if "vocals" in stems:
            return [stems["vocals"]]
        
        # Fallback to other
        if "other" in stems:
            return [stems["other"]]
            
        return []

    def resolve_backing_stems(self, stems: Dict[str, Path], solo_paths: List[Path]) -> List[Path]:
        """
        Ensure backing is everything EXCEPT the chosen solo stem.
        """
        return super().resolve_backing_stems(stems, solo_paths)
