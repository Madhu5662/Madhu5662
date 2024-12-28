import numpy as np
import torch
import argparse

from collections import Counter
from sklearn.preprocessing import LabelEncoder
from librosa.core import load
from librosa.feature import melspectrogram

from model import genreNet
from config import MODELPATH, GENRES

import warnings

warnings.filterwarnings("ignore")


def load_audio(audio_path, sr=22050):
    """Load audio file and return audio time-series and sample rate."""
    try:
        y, sr = load(audio_path, mono=True, sr=sr)
        return y, sr
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {e}")


def get_spectrogram_chunks(y, sr, chunk_size=128):
    """Convert audio time-series to mel spectrogram and split into chunks."""
    S = melspectrogram(y, sr).T
    S = S[: -(S.shape[0] % chunk_size)]  # Trim to ensure divisibility
    num_chunks = S.shape[0] // chunk_size
    return np.split(S, num_chunks)


def classify_chunks(model, data_chunks, label_encoder, device, threshold=0.5):
    """Classify each spectrogram chunk and return predicted genres."""
    genres = []
    with torch.no_grad():  # Disable gradients for inference
        for chunk in data_chunks:
            data = torch.FloatTensor(chunk).view(1, 1, 128, 128).to(device)
            preds = model(data)
            pred_val, pred_index = torch.max(preds, dim=1)
            pred_val = torch.exp(pred_val).item()
            pred_index = pred_index.item()
            if pred_val >= threshold:
                genres.append(label_encoder.inverse_transform([pred_index])[0])
    return genres


def display_genre_distribution(genres):
    """Display the percentage distribution of predicted genres."""
    if not genres:
        print("No genres confidently detected.")
        return

    genre_counts = Counter(genres)
    total = sum(genre_counts.values())
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)

    print("Genre Distribution:")
    for genre, count in sorted_genres:
        percentage = (count / total) * 100
        print(f"{genre:>10}: {percentage:.2f}%")


def main(audio_path, device):
    # Load label encoder
    label_encoder = LabelEncoder().fit(GENRES)

    # Load trained model
    model = genreNet().to(device)
    model.load_state_dict(torch.load(MODELPATH, map_location=device))
    model.eval()  # Set model to evaluation mode

    # Load audio and process
    y, sr = load_audio(audio_path)
    data_chunks = get_spectrogram_chunks(y, sr)

    # Classify audio chunks
    genres = classify_chunks(model, data_chunks, label_encoder, device)

    # Display results
    display_genre_distribution(genres)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Genre Classification Script")
    parser.add_argument("audiopath", type=str, help="Path to the audio file")
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run the model on"
    )
    args = parser.parse_args()

    try:
        main(args.audiopath, args.device)
    except Exception as e:
        print(f"Error: {e}")
