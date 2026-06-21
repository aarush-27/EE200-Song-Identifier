import os
import librosa
import matplotlib.pyplot as plt

from fingerprint import plot_constellation

os.makedirs(
    "constellation_images",
    exist_ok=True
)

for filename in os.listdir("songs"):

    if not filename.endswith(".mp3"):
        continue

    print("Processing:", filename)

    song_path = os.path.join(
        "songs",
        filename
    )

    y, sr = librosa.load(
        song_path,
        sr=None
    )

    fig = plot_constellation(
        y,
        sr
    )

    output_name = (
        os.path.splitext(filename)[0]
        + ".png"
    )

    fig.savefig(
        os.path.join(
            "constellation_images",
            output_name
        ),
        bbox_inches="tight"
    )

    plt.close(fig)

print("Done.")