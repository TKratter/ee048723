import numpy as np
from scipy.io import wavfile


def load_wav_file(filename: str) -> np.ndarray:
    sample_rate, data = wavfile.read(filename)
    return data

def dtw(wav1: np.ndarray, wav2: np.ndarray) -> float:
    pass


def compare_wav_files_by_metric(filename1: str, filename2: str, metric: str) -> float:
    wav1 = load_wav_file(filename1)
    wav2 = load_wav_file(filename2)
    if metric == 'euclidean':
        return np.linalg.norm(wav1 - wav2)
    elif metric == 'dtw':
        return dtw(wav1, wav2)
