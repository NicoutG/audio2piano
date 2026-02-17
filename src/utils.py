import matplotlib.pyplot as plt
import numpy as np

def show_spectrogram(mat, title, y_label):
    if mat.shape[0] < mat.shape[1]:
        mat = mat.T
    plt.figure(figsize=(14, 6))
    plt.imshow(mat.T, aspect="auto", origin="lower", cmap="magma")
    plt.xlabel("Time frames")
    plt.ylabel(y_label)
    plt.title(title)
    plt.colorbar()
    plt.show()

def show_piano_roll(mat, title="Piano Roll"):
    if mat.shape[1] != 88:
        mat = mat.T
    plt.figure(figsize=(14, 6))
    plt.imshow(mat.T, aspect="auto", origin="lower", cmap="gray_r")
    plt.xlabel("Time frames")
    plt.ylabel("MIDI pitch (21–108)")
    plt.title(title)
    plt.colorbar()
    plt.show()

def midi_to_sustain_roll(notes, total_steps, hop_sec=0.05):
    roll = np.zeros((88, total_steps), dtype=np.float32)

    for note in notes:
        pitch = note["pitch"] - 21
        if not (0 <= pitch < 88):
            continue

        start = int(note["start"] / hop_sec)
        end   = int(note["end"]   / hop_sec)

        if start >= total_steps:
            continue

        end = max(start + 1, end)
        end = min(end, total_steps)

        duration = end - start

        for i, t in enumerate(range(start, end)):
            alpha = i / duration

            value = 1.0 - 0.5 * alpha

            roll[pitch, t] = max(roll[pitch, t], value)

    return roll