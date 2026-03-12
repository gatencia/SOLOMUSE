import logging
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from solomuse_data.dataset_adapters.base import DatasetAdapter, Track
from solomuse_data.config import PipelineConfig

logger = logging.getLogger(__name__)

class SlakhAdapter(DatasetAdapter):
    """
    Adapter for Slakh2100 dataset.
    Structure:
      Round1/
        Track00001/
          stems/   (or sometimes just stems in root?)
            S01.wav
            S02.wav
          metadata.yaml (optional)
          mix.wav
    """
    
    def __init__(self, root: Path, cfg: PipelineConfig):
        super().__init__(root, cfg)
        
    def list_tracks(self) -> List[Track]:
        """
        Walk directories to find tracks.
        A directory is a track if it contains 'stems' subdirectory or has multiple wavs.
        Slakh usually has `TrackXXXXX` folders.
        """
        tracks = []
        if not self.root.exists():
            logger.warning(f"Slakh root {self.root} does not exist.")
            return []

        # Recurse or just list top level? 
        # Slakh often comes as `slakh2100_train/Track0001`.
        # Let's walk and look for directories that look like tracks.
        # A simple heuristic: any dir containing a 'stems' subdir or 'mix.wav'.
        
        for path in self.root.rglob("*"):
             if path.is_dir():
                 # check if it is a track dir
                 if (path / "stems").exists() and (path / "stems").is_dir():
                     track_id = path.name
                     tracks.append(Track(dataset="slakh", track_id=track_id, root=path))
                 elif (path / "mix.wav").exists() or (path / "mixture.wav").exists() or (path / "mix.flac").exists():
                     # Fallback if stems folder structure is flat? Slakh usually has stems dir
                     track_id = path.name
                     tracks.append(Track(dataset="slakh", track_id=track_id, root=path))
                     
        return tracks

    def list_stems(self, track: Track) -> Dict[str, Path]:
        stems = {}
        stems_dir = track.root / "stems"
        
        # Determine metadata path
        meta_path = track.root / "metadata.yaml"
        stem_meta = {}
        
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    data = yaml.safe_load(f)
                    stem_meta = data.get("stems", {})
            except Exception as e:
                logger.warning(f"Failed to read metadata for {track.track_id}: {e}")
        
        if stems_dir.exists():
            # Standard Slakh - support multiple formats
            for f in stems_dir.iterdir():
                if f.suffix.lower() in [".wav", ".flac", ".mp3"]:
                    # Use filename stem (S00)
                    file_stem = f.stem
                    
                    # Look up instrument class
                    # Metadata keys are usually S00, S01... match file_stem
                    # Append instrument class to name for policy matching
                    # e.g. "S00_Guitar"
                    
                    name = file_stem
                    if file_stem in stem_meta:
                        inst_class = stem_meta[file_stem].get("inst_class", "")
                        if inst_class:
                            name = f"{file_stem}_{inst_class}"
                            
                    stems[name] = f
        else:
             # Flat structure fallback
              for f in track.root.iterdir():
                  if f.suffix.lower() in [".wav", ".flac", ".mp3"] and f.stem.lower() not in ["mix", "mixture"]:
                     stems[f.stem] = f
                     
        return stems

    def get_mix_path(self, track: Track) -> Optional[Path]:
        exts = [".wav", ".flac", ".mp3"]
        names = ["mix", "mixture"]
        for n in names:
            for e in exts:
                p = track.root / f"{n}{e}"
                if p.exists():
                    return p
        return None
    
    def get_metadata(self, track: Track) -> Dict:
        meta = super().get_metadata(track)
        meta["license"] = "CC BY 4.0"
        return meta
