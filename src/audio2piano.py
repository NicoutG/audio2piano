from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import mido

NB_BINS = 288
HOP_SEC = 0.05
SR = 22050
THRESHOLD = 0.6

class Audio2Piano(nn.Module):
    def __init__(
        self,
        weights_path="weights/model_weights.pth",
        device=None,
        n_bins=NB_BINS,
        cnn_hidden=192,
        rnn_hidden=384,
        n_notes=88,
        dropout=0.3,
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # CNN
        self.cnn1 = nn.Sequential(
            nn.Conv1d(n_bins, cnn_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_hidden),
            nn.ReLU(),
        )

        self.cnn2 = nn.Sequential(
            nn.Conv1d(cnn_hidden, cnn_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_hidden),
            nn.ReLU(),
        )

        self.cnn3 = nn.Sequential(
            nn.Conv1d(cnn_hidden, cnn_hidden, kernel_size=1),
            nn.BatchNorm1d(cnn_hidden),
            nn.ReLU(),
        )

        self.cnn_dropout = nn.Dropout(dropout)

        # GRU
        self.gru = nn.GRU(
            input_size=cnn_hidden * 2,
            hidden_size=rnn_hidden,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Projection
        self.proj_onset = nn.Sequential(
            nn.Linear(rnn_hidden * 2, rnn_hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden * 2, rnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, n_notes)
        )

        self.proj_sustain = nn.Sequential(
            nn.Linear(rnn_hidden * 2, rnn_hidden * 2),
            nn.ReLU(),
            nn.Linear(rnn_hidden * 2, rnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, n_notes)
        )

        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location='cpu')
            self.load_state_dict(state_dict)

    def forward(self, x):
        # (B, T, F) → (B, F, T)
        x = x.transpose(1, 2)

        # CNN
        h1 = self.cnn1(x)
        h2 = self.cnn2(h1)
        h = h1 + h2
        h = self.cnn3(h)
        h = self.cnn_dropout(h)

        # (B, C, T) → (B, T, C)
        h = h.transpose(1, 2)

        # Delta
        dh = h[:, 1:] - h[:, :-1]
        dh = F.pad(dh, (0, 0, 1, 0))

        h = torch.cat([h, dh], dim=-1)  # (B, T, 2C)

        # GRU
        h, _ = self.gru(h)

        # Projection
        out_onset = self.proj_onset(h)
        out_sustain = self.proj_sustain(h)

        return out_onset, out_sustain  # logits
    
    def load_wav(self, wav_path, sr=SR):
        samples, sr = librosa.load(wav_path, sr=sr, mono=True)
        return samples, sr
    
    def wav_to_mel(self,
        samples,
        sr=SR,
        hop_sec=HOP_SEC,
        n_mels=NB_BINS,
        fmin=30.0,
        fmax=8000,
        alpha=0.8
    ):
        if fmax is None:
            fmax = sr / 2

        hop_length = int(sr * hop_sec)

        mel = librosa.feature.melspectrogram(
            y=samples,
            sr=sr,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            power=2.0
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean(axis=1, keepdims=True)) / \
                (mel_db.std(axis=1, keepdims=True) + 1e-6)
        
        mel_db2 = np.power(mel, 0.3)
        mel_db2 = (mel_db2 - mel_db2.mean()) / (mel_db2.std() + 1e-6)

        mel_mix = alpha * mel_db + (1 - alpha) * mel_db2

        return torch.tensor(mel_mix, dtype=torch.float32)
    
    @torch.no_grad()
    def predict_midi(
        self,
        samples,
        sr=SR,
        threshold=THRESHOLD,
        hop_sec=HOP_SEC,
        decay_base=0.5,
        decay_growth=2.0,
        max_duration=10.0,
        min_onset_interval=0.2,
        ticks_per_beat=480,
        tempo=500_000
    ):
        self.eval()

        mel = self.wav_to_mel(samples, sr=sr, hop_sec=hop_sec)  # (F, T)
        mel = mel.transpose(0, 1).unsqueeze(0).to(self.device)  # (1, T, F)

        out_onset, out_sustain = self.forward(mel)  # (1, T, 88)
        onset_probs = torch.sigmoid(out_onset[0]).cpu().numpy()
        sustain_probs = torch.sigmoid(out_sustain[0]).cpu().numpy()

        notes = note_matrices_to_notes(
            onset_probs,
            sustain_probs,
            hop_sec=hop_sec,
            onset_threshold=threshold,
            decay_base=decay_base,
            decay_growth=decay_growth,
            max_duration=max_duration,
            min_onset_interval=min_onset_interval
        )

        total_duration = len(samples) / sr
        midi = create_midi_from_notes(
            notes,
            total_duration=total_duration,
            ticks_per_beat=ticks_per_beat,
            tempo=tempo
        )

        return midi
    
    def wav_to_midi_file(self, input_wav, output_midi, sr=SR, threshold=THRESHOLD):
        samples, sr = self.load_wav(input_wav, sr)

        midi = self.predict_midi(samples, sr=sr, threshold=threshold)

        output_midi = Path(output_midi)
        output_midi.parent.mkdir(parents=True, exist_ok=True)
        midi.save(str(output_midi))

        return midi
    
    def wav_to_midi_folder(self, input_folder, output_folder, sr=SR, threshold=THRESHOLD, recursive=False):
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        if recursive:
            wav_files = list(input_folder.rglob("*.wav"))
        else:
            wav_files = list(input_folder.glob("*.wav"))

        if not wav_files:
            print(f"[WARN] No WAV files found in {input_folder}")
            return

        print(f"[INFO] Found {len(wav_files)} WAV files. Converting to MIDI...")

        for wav_path in wav_files:
            rel_path = wav_path.relative_to(input_folder).with_suffix(".mid")
            midi_path = output_folder / rel_path
            midi_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Converting: {wav_path} → {midi_path}")
            self.wav_to_midi_file(str(wav_path), str(midi_path), sr=sr, threshold=threshold)

        print("[OK] All files processed.")

def note_matrices_to_notes(
    onset_matrix,
    sustain_matrix,
    hop_sec=HOP_SEC,
    onset_threshold=THRESHOLD,
    decay_base=0.4,
    max_duration=10.0,
    decay_growth=2.0,
    min_onset_interval=0.2
):

    T, N = sustain_matrix.shape
    notes = []
    active_notes = {}
    last_onset_time = {}

    for t in range(T):
        time = t * hop_sec

        for p in range(N):
            onset_val = onset_matrix[t, p]
            sustain_val = sustain_matrix[t, p]

            if onset_val >= onset_threshold:

                if p in last_onset_time:
                    if time - last_onset_time[p] < min_onset_interval:
                        continue

                last_onset_time[p] = time

                if p in active_notes:
                    start = active_notes[p]["start_time"]
                    notes.append({
                        "pitch": p + 21,
                        "start": start,
                        "end": time
                    })

                active_notes[p] = {
                    "start_time": time,
                    "power": 1.0
                }
                continue

            if p in active_notes:

                start_time = active_notes[p]["start_time"]
                duration = time - start_time

                duration_ratio = min(duration / max_duration, 2.0)
                dynamic_decay = decay_base * (1 + decay_growth * duration_ratio)

                new_power = (
                    active_notes[p]["power"]
                    - dynamic_decay
                    + sustain_val ** 2
                )

                active_notes[p]["power"] = min(1.0, new_power)

                if active_notes[p]["power"] <= 0 or duration >= max_duration:
                    notes.append({
                        "pitch": p + 21,
                        "start": start_time,
                        "end": time
                    })
                    del active_notes[p]

    final_time = T * hop_sec
    for p, note in active_notes.items():
        notes.append({
            "pitch": p + 21,
            "start": note["start_time"],
            "end": final_time
        })

    return notes

def create_midi_from_notes(notes, total_duration, ticks_per_beat=480, tempo=500_000):
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    ticks_per_second = ticks_per_beat * 1_000_000 / tempo
    events = []

    for n in notes:
        on_tick = int(n["start"] * ticks_per_second)
        off_tick = int(n["end"] * ticks_per_second)
        events.append((on_tick, "note_on", n["pitch"], 100))
        events.append((off_tick, "note_off", n["pitch"], 0))

    events.sort(key=lambda x: x[0])
    last_tick = 0
    for tick, typ, pitch, vel in events:
        track.append(
            mido.Message(typ, note=pitch, velocity=vel, time=max(1, tick - last_tick))
        )
        last_tick = tick

    end_tick = int(total_duration * ticks_per_second)
    track.append(mido.MetaMessage("end_of_track", time=max(1, end_tick - last_tick)))
    return mid
