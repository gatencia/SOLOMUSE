import abc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
from solomuse_data.config import PipelineConfig

@dataclass
class Track:
    dataset: str
    track_id: str
    root: Path

def normalize_stem_name(name: str) -> str:
    """Normalize stem name for matching policy."""
    return name.lower().replace(" ", "_").replace("-", "_")

class DatasetAdapter(abc.ABC):
    def __init__(self, root: Path, cfg: PipelineConfig):
        self.root = root
        self.cfg = cfg
        # Supported keywords for 'lead_any' policy
        self.SOLO_KEYWORDS = {
            "lead", "melody", "solo", "vocal", "vox", "singer", "lead_guitar", "guitar_lead",
            "guitar", "piano", "keys", "synthesizer", "synth", "strings", "violin", "cello",
            "sax", "saxophone", "brass", "trumpet", "flute", "wind"
        }

    @abc.abstractmethod
    def list_tracks(self) -> List[Track]:
        """List all tracks in the dataset."""
        pass

    @abc.abstractmethod
    def list_stems(self, track: Track) -> Dict[str, Path]:
        """
        List all available stems for a track.
        Returns:
            dict mapping stem_name -> stem_path
        """
        pass

    @abc.abstractmethod
    def get_mix_path(self, track: Track) -> Optional[Path]:
        """Return path to mixture file if available."""
        pass

    def get_metadata(self, track: Track) -> Dict:
        """Return dataset-specific metadata."""
        return {
            "dataset": self.cfg.dataset_roots.get(track.dataset, "unknown"),
            "track_id": track.track_id
        }

    def resolve_solo_stems(self, stems: Dict[str, Path]) -> List[Path]:
        """
        Identify solo stems based on configuration policy.
        """
        if self.cfg.solo_stem_policy != "lead_any":
            # Currently only lead_any is supported/locked
            return []

        solo_paths = []
        for name, path in stems.items():
            norm_name = normalize_stem_name(name)
            
            # Check keywords
            # "lead_guitar" matches because "lead" is in keywords? 
            # Or are keywords substrings? 
            # Prompt: "Treat stems whose names contain any of: [list]"
            
            for kw in self.SOLO_KEYWORDS:
                if kw in norm_name:
                    solo_paths.append(path)
                    break 

        return solo_paths

    def resolve_backing_stems(self, stems: Dict[str, Path], solo_paths: List[Path]) -> List[Path]:
        """
        Identify backing stems (complement of solo stems).
        """
        backing_paths = []
        solo_set = set(solo_paths)
        
        for path in stems.values():
            if path not in solo_set:
                backing_paths.append(path)
                
        return backing_paths
