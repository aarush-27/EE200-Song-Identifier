import streamlit as st
import librosa

from fingerprint import generate_hashes
from matcher import identify_song

from fingerprint import (
    generate_hashes,
    plot_spectrogram,
    plot_constellation
)

from matcher import (
    identify_song,
    plot_histogram
)

st.title("EE200 Song Identifier")

tab1, tab2, tab3 = st.tabs(
    [
        "Library",
        "Single Clip",
        "Batch Mode"
    ]
)

# ==========================
# TAB 1
# ==========================

with tab1:

    import os

    st.header("Indexed Song Library")

    songs = sorted(
        [
            os.path.splitext(f)[0]
            for f in os.listdir("songs")
            if f.endswith(".mp3")
        ]
    )

    st.write(f"Total Songs: {len(songs)}")

    st.divider()

    cols = st.columns(3)

    for i, song in enumerate(songs):

        image_path = (
            f"constellation_images/{song}.png"
        )

        with cols[i % 3]:

            if os.path.exists(image_path):

                st.image(
                    image_path,
                    use_container_width=True
                )

            else:

                st.image(
                    "https://via.placeholder.com/300x200?text=No+Image",
                    use_container_width=True
                )

            st.caption(song)

# ==========================
# TAB 2
# ==========================

with tab2:

    uploaded_file = st.file_uploader(
        "Upload an audio clip",
        type=["mp3", "wav"]
    )

    if uploaded_file is not None:

        y, sr = librosa.load(
            uploaded_file,
            sr=None
        )

        with st.spinner(
            "Identifying song..."
        ):

            query_hashes = generate_hashes(
                y,
                sr
            )

            result = identify_song(
                query_hashes
            )

        if result is not None:

            song, score, offsets = result

            st.success(
                f"Predicted Song: {song}"
            )

            col1, col2 = st.columns([3, 1])

            with col1:

                st.subheader("Matched Song")

                st.success(song)

            with col2:

                st.subheader("Votes")

                st.metric(
                    label="",
                    value=score
                )

            st.divider()

            st.subheader(
                "Spectrogram"
            )

            spec_fig = plot_spectrogram(
                y,
                sr
            )

            st.pyplot(
                spec_fig,
                use_container_width=True
            )

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "Constellation Map"
                )

                const_fig = plot_constellation(
                    y,
                    sr
                )

                st.pyplot(
                    const_fig,
                    use_container_width=True
                )

            with col2:

                st.subheader(
                    "Offset Histogram"
                )

                hist_fig = plot_histogram(
                    offsets,
                    song
                )

                st.pyplot(
                    hist_fig,
                    use_container_width=True
                )

        else:

            st.error(
                "No matching song found."
            )

# ==========================
# TAB 3
# ==========================

with tab3:

    st.header("Batch Identification")

    uploaded_files = st.file_uploader(
        "Upload multiple audio clips",
        type=["mp3", "wav"],
        accept_multiple_files=True
    )

    if uploaded_files:

        import pandas as pd

        results = []

        progress = st.progress(0)

        for i, file in enumerate(uploaded_files):

            y, sr = librosa.load(
                file,
                sr=None
            )

            query_hashes = generate_hashes(
                y,
                sr
            )

            result = identify_song(
                query_hashes
            )

            if result is not None:

                song, score, offsets = result

            else:

                song = "Unknown"

            filename = file.name

            prediction = song

            results.append(
                {
                    "filename": filename,
                    "prediction": prediction
                }
            )

            progress.progress(
                (i + 1) / len(uploaded_files)
            )

        df = pd.DataFrame(results)

        st.subheader("Predictions")

        st.dataframe(df)

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download results.csv",
            csv,
            "results.csv",
            "text/csv"
        )