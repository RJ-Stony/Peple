# --- 1. Load & Preprocess Audio Data ---
import librosa

def load_audio(filepath: str, sr: int = None):
    """
    Load an audio file and return the audio time series and sampling rate.

    Args:
        filepath (str): Path to the audio (.wav) file.
        sr (int, optional): Target sampling rate. If None, uses the file's original rate.

    Returns:
        Tuple[np.ndarray, int]: (audio_array, sampling_rate)
    """
    # librosa.load returns float32 numpy array and sampling rate
    audio, sampling_rate = librosa.load(filepath, sr=sr)
    return audio, sampling_rate
