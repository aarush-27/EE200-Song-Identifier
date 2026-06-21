import os
import pickle
import librosa

from fingerprint import generate_hashes

DATABASE = {}

SONG_FOLDER = "songs"

for filename in os.listdir(SONG_FOLDER):

    if not filename.endswith(".mp3"):
        continue

    path = os.path.join(
        SONG_FOLDER,
        filename
    )

    print(
        "Processing:",
        filename
    )

    y, sr = librosa.load(
        path,
        sr=None
    )

    hashes = generate_hashes(
        y,
        sr
    )

    song_name = os.path.splitext(
        filename
    )[0]

    for f1, f2, dt, t1 in hashes:

        hash_key = (
            round(f1),
            round(f2),
            int(dt)
        )

        if hash_key not in DATABASE:
            DATABASE[hash_key] = []

        DATABASE[hash_key].append(
            (
                song_name,
                int(t1)
            )
        )
with open(
    "database.pkl",
    "wb"
) as f:

    pickle.dump(
        DATABASE,
        f
    )

print(
    "Database saved."
)

print(
    "Unique hashes:",
    len(DATABASE)
)