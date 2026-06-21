import librosa
import librosa.display
import numpy as np
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt


def generate_hashes(signal, sr):

    D = librosa.stft(
        signal,
        n_fft=2048,
        hop_length=512
    )

    S = np.abs(D)

    S_db = librosa.amplitude_to_db(
        S,
        ref=np.max
    )

    local_max = maximum_filter(
        S_db,
        size=20
    )

    peaks = (
        (S_db == local_max)
        &
        (S_db > -30)
    )

    freq_idx, time_idx = np.where(peaks)

    peak_points = list(
        zip(time_idx, freq_idx)
    )

    peak_points.sort()

    freqs = librosa.fft_frequencies(
        sr=sr,
        n_fft=2048
    )

    hashes = []

    for i in range(
        min(10000, len(peak_points)-5)
    ):

        t1, fidx1 = peak_points[i]

        for j in range(1, 6):

            t2, fidx2 = peak_points[i+j]

            f1 = freqs[fidx1]
            f2 = freqs[fidx2]

            dt = t2 - t1

            hashes.append(
                (
                    f1,
                    f2,
                    dt,
                    t1
                )
            )

    return hashes

def plot_spectrogram(signal, sr):

    D = librosa.stft(
        signal,
        n_fft=2048,
        hop_length=512
    )

    S_db = librosa.amplitude_to_db(
        np.abs(D),
        ref=np.max
    )

    fig, ax = plt.subplots(
        figsize=(10,4)
    )


    librosa.display.specshow(
        S_db,
        sr=sr,
        x_axis='time',
        y_axis='hz',
        ax=ax
    )

    ax.set_title(
        "Spectrogram"
    )

    return fig

def plot_constellation(
    signal,
    sr
):

    D = librosa.stft(
        signal,
        n_fft=2048,
        hop_length=512
    )

    S = np.abs(D)

    S_db = librosa.amplitude_to_db(
        S,
        ref=np.max
    )

    local_max = maximum_filter(
        S_db,
        size=20
    )

    peaks = (
        (S_db == local_max)
        &
        (S_db > -30)
    )

    freq_idx, time_idx = np.where(
        peaks
    )

    freqs = librosa.fft_frequencies(
        sr=sr,
        n_fft=2048
    )

    fig, ax = plt.subplots(
        figsize=(10,4)
    )

    ax.scatter(
        time_idx,
        freqs[freq_idx],
        s=1
    )

    ax.set_title(
        "Constellation Map"
    )

    return fig