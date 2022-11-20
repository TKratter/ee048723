import numpy as np
from scipy.io import wavfile
import numba


def load_wav_file(filename: str) -> np.ndarray:
    sample_rate, data = wavfile.read(filename)
    return data


@numba.jit(nopython=True)
def dtw(wav1: np.ndarray, wav2: np.ndarray) -> float:
    dtw_matrix = np.ones((len(wav1) + 1, len(wav2) + 1)) * np.inf
    dtw_matrix[0, 0] = 0
    for i in range(1, len(wav1) + 1):
        for j in range(1, len(wav2) + 1):
            dist = np.abs(wav1[i - 1] - wav2[j - 1])
            dtw_matrix[i, j] = dist + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )
    return dtw_matrix[-1, -1]


def compare_wav_files_by_metric(filename1: str, filename2: str, metric: str) -> float:
    wav1 = load_wav_file(filename1)
    wav2 = load_wav_file(filename2)
    if metric == 'euclidean':
        return np.linalg.norm(wav1 - wav2)
    elif metric == 'dtw':
        return dtw(wav1, wav2)


def classify_wav_by_metric(filename: str, metric: str) -> int:
    distances = []
    for i in range(1, 4):
        distances.append(compare_wav_files_by_metric(filename, f'sample{i}.wav', metric))
    return np.argmin(distances) + 1
