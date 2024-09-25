import librosa
import librosa.display
import matplotlib.pylab as plt
import seaborn as sns
import os
import pickle
import pandas as pd
import numpy as np
import cv2
import soundfile as sf

from dotenv import load_dotenv


def get_spectrogram(audio, fft_size=2048, hop_size=None, window_size=None, to_db=True):

    if not window_size:
        window_size = fft_size

    if not hop_size:
        hop_size = window_size // 4

    D = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size, win_length=window_size)
    D = np.abs(D)

    S_db = librosa.amplitude_to_db(D, ref=np.max)
    if to_db:
        return S_db
    else:
        return D


def get_chromagram(
    audio,
    sr,
    fft_size=2048,
    hop_size=None,
    window_size=None,
    power_spectrum=False,
    target_len=None,
):
    chroma = None
    if not power_spectrum:
        # Usar el espectro de magnitud
        D = get_spectrogram(
            audio=audio,
            fft_size=fft_size,
            hop_size=hop_size,
            window_size=window_size,
            to_db=False,
        )
        chroma = librosa.feature.chroma_stft(S=D, sr=sr)
    else:
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

    if target_len:
        chroma = adjust_spectrogram(chroma, target_len)

    return chroma


def show_spectrogram(spectrogram, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(spectrogram, x_axis="time", y_axis="log", ax=ax)

    ax.set_title(f"{title}", fontsize=(20))
    fig.colorbar(img, ax=ax, format=f"%0.2f")

    plt.show()


def show_chromagram(chroma, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(chroma, x_axis="time", y_axis="chroma", ax=ax)
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


def redimensionar(img, resize):
    img_resized = cv2.resize(img, resize)

    return img_resized


def resize_spectrogramas(spectrograms, resize):
    spec_res = []

    for img in spectrograms:
        img_res = redimensionar(img, resize)
        spec_res.append(img_res)

    return np.array(spec_res)


def stretching(audio, tst):
    return librosa.effects.time_stretch(audio, rate=tst)


def shifting(audio, tsh, sr):
    shift_sample = int(tsh * sr)
    return np.roll(audio, shift_sample)


def data_augmentation(
    audio_files, labels, length, tsh=5, tst_list=[0.81, 0.93, 1.07, 1.23], sr=None
):
    if not load_dotenv:
        print(f"No se encontro archivo de variables de entorno")
        return None

    AUDIO_DIR = os.getenv("PMEMO_AUDIO")
    if not os.path.exists(AUDIO_DIR):
        print(f"La ruta de audios eespecificada no existe")
        return None

    AUG_DIR = os.path.join(AUDIO_DIR, "augmented")
    if not os.path.exists(AUG_DIR):
        print(f"Creando carpeta {os.path.basename(AUG_DIR)} . . .")
        os.mkdir(AUG_DIR)

    if not sr:
        sr = 22050

    input_len = sr * length

    music_ids = []
    valence = []
    arousal = []
    is_augmented = []
    comments = []
    audio_dir = []

    for r, audio in enumerate(audio_files):
        audio_name = os.path.basename(audio)
        current_id = labels.iloc[r].musicId
        current_arousal = labels.iloc[r]["Arousal"]
        current_valence = labels.iloc[r]["Valence"]

        music_ids.append(current_id)
        valence.append(current_valence)
        arousal.append(current_arousal)
        is_augmented.append(False)
        comments.append("Source file")
        audio_dir.append(audio)

        y, sr = librosa.load(audio, offset=0, duration=length)
        for i, tst in enumerate(tst_list):
            aug_name = f"{audio_name.split('.')[0]}_tst{i}.mp3"
            aug_dir = os.path.join(AUG_DIR, aug_name)
            aug_audio = stretching(y, tst)

            if len(aug_audio) > input_len:
                aug_audio = aug_audio[:input_len]

            sf.write(aug_dir, aug_audio, sr)
            music_ids.append(current_id)
            valence.append(current_valence)
            arousal.append(current_arousal)
            is_augmented.append(True)
            comments.append(f"Time stretching {tst} seconds")
            audio_dir.append(aug_dir)

        aug_name = f"{audio_name.split('.')[0]}_tsh{tsh}.mp3"
        aug_dir = os.path.join(AUG_DIR, aug_name)
        aug_audio = shifting(y, tsh, sr)

        sf.write(aug_dir, aug_audio, sr)
        music_ids.append(current_id)
        valence.append(current_valence)
        arousal.append(current_arousal)
        is_augmented.append(True)
        comments.append(f"Time shifting {tsh} seconds")
        audio_dir.append(aug_dir)

    new_cols = {
        "musicId": music_ids,
        "Arousal(mean)": arousal,
        "Valence(mean)": valence,
        "isAugmented": is_augmented,
        "info": comments,
        "audioDir": audio_dir,
    }

    return pd.DataFrame(new_cols)
