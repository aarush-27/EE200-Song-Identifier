from collections import Counter
import pickle
import gzip

with gzip.open(
    "database.pkl.gz",
    "rb"
) as f:

    DATABASE = pickle.load(f)


def identify_song(query_hashes):

    votes = Counter()

    offsets_by_song = {}

    for f1, f2, dt, t1_query in query_hashes:

        hash_key = (
            round(f1),
            round(f2),
            int(dt)
        )

        if hash_key not in DATABASE:
            continue

        for song_name, t1_database in DATABASE[hash_key]:

            offset = t1_database - t1_query

            votes[(song_name, offset)] += 1

            if song_name not in offsets_by_song:
                offsets_by_song[song_name] = []

            offsets_by_song[song_name].append(offset)

    if len(votes) == 0:
        return None

    best_match = votes.most_common(1)[0]

    (best_song, best_offset), score = best_match

    return (
        best_song,
        score,
        offsets_by_song[best_song]
    )

import matplotlib.pyplot as plt

def plot_histogram(offsets, song):

    fig, ax = plt.subplots(figsize=(10,4))

    ax.hist(
        offsets,
        bins=100
    )

    ax.set_title(
        f"Offset Histogram: {song}"
    )

    ax.set_xlabel("Offset")
    ax.set_ylabel("Count")

    return fig
