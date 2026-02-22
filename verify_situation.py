import numpy as np
import librosa
from solomuse_model.situation.extract import extract_situation_v1
from solomuse_model.situation.vectorize import vectorize_situation_v1
from solomuse_data.config import PipelineConfig

def test_manual():
    print("Testing manual situation extraction...")
    cfg = PipelineConfig(
        output_root="./test_out",
        dataset_roots={"mock": "./data/raw/mock"},
    )
    
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    audio = audio[:, np.newaxis]
    
    print("Extracting features...")
    features = extract_situation_v1(audio, sr, cfg)
    print(f"Features: tempo={features['tempo_bpm']}, loudness={features['loudness_lufs']}, rms={features['rms_mean']}")
    
    print("Vectorizing...")
    vector = vectorize_situation_v1(features)
    print(f"Vector shape: {vector.shape}, dtype: {vector.dtype}")
    print(f"First 10 values: {vector[:10]}")
    
    assert vector.shape == (32,)
    print("Success!")

if __name__ == "__main__":
    test_manual()
