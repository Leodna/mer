import librosa
import librosa.display
import matplotlib.pylab as plt
import seaborn as sns
import os
import pickle
import pandas as pd
import numpy as np

from dotenv import load_dotenv


def get_spectrogram(audio, fft_size=2048, hop_size=None, window_size=None):

    if not window_size:
        window_size = fft_size

    if not hop_size:
        hop_size = window_size // 4

    D = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size, win_length=window_size)
    D = np.abs(D)
    S_db = librosa.amplitude_to_db(D, ref=np.max)

    return S_db


def show_spectrogram(spectrogram, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(spectrogram, x_axis="time", y_axis="log", ax=ax)

    ax.set_title(f"{title}", fontsize=(20))
    fig.colorbar(img, ax=ax, format=f"%0.2f")

    plt.show()


def show_distribution(data, title):
    sns.histplot(data, kde=True).set_title(f"{title}")

    plt.show()


def save_spectrograms(spectrograms, filename, dir):
    if not load_dotenv():
        print("No se encontró archivo de variables de entorno")
        return None

    ASSETS_DIR = os.getenv("ASSETS_DIR")
    spect_file = f"{filename}.pkl"
    spect_dir = os.path.join(ASSETS_DIR, spect_file)

    with open(spect_dir, "wb") as f:
        pickle.dump(spectrograms, f)
    print("Espectrogramas guardados exitosamente")

    return spect_file


def load_spectrograms(filename):
    if not load_dotenv():
        print("No se encontró archivo de variables de entorno")
        return None

    ASSETS_DIR = os.getenv("ASSETS_DIR")
    spect_file = f"{filename}.pkl"
    spect_dir = os.path.join(ASSETS_DIR, spect_file)

    # Comprobar que existe el archivo
    if not os.path.exists(spect_dir):
        print(f"Archivo {filename} no existe en la ruta predefinida")
        return None

    with open(spect_dir, "rb") as f:
        loaded_specs = pickle.load(f)

    return loaded_specs


def adjust_spectrogram(spectrogram, target_len):
    current_len = spectrogram.shape[1]

    adj_spec = librosa.util.fix_length(spectrogram, size=target_len)

    return adj_spec


def show_audio_signal(audio, title, color="#8839ef"):
    pd.Series(audio).plot(figsize=(10, 5), lw=1, title=f"{title}", color=color)
    plt.show()
