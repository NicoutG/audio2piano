import torch
import numpy as np
from pathlib import Path
import numpy as np

from audio2piano import *
from musicPredictor import MusicPredictor

ALPHA_AUDIO = 0.8
THRESHOLD_PRED = 0.5

class Audio2PianoMusicPrior:

    def __init__(
        self,
        audio_weights="weights/audio2piano_weights.pth",
        prior_weights="weights/musicPredictor_weights.pth",
        device=None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.audio_model = Audio2Piano(
            weights_path=audio_weights,
            device=device
        ).to(self.device)

        self.prior_model = MusicPredictor(
            weights_path=prior_weights,
            device=device
        ).to(self.device)

        self.audio_model.eval()
        self.prior_model.eval()

    def forward(self, x, threshold_pred=THRESHOLD, alpha_audio=ALPHA_AUDIO):
        out_onset, out_sustain = self.audio_model(x)

        onset_logits = out_onset[0]
        sustain_logits = out_sustain[0]

        return self._apply_prior(
            onset_logits,
            sustain_logits,
            threshold_pred=threshold_pred,
            alpha_audio = alpha_audio
        )
    
    def load_wav(self, wav_path, sr=SR):
        return self.audio_model.load_wav(wav_path=wav_path, sr=sr)
    
    def wav_to_mel(self,
        samples,
        sr=SR,
        hop_sec=HOP_SEC,
        n_mels=NB_BINS,
        fmin=30.0,
        fmax=8000,
        alpha=0.8
    ):
        return self.audio_model.wav_to_mel(samples=samples, sr=sr, hop_sec=hop_sec, n_mels=n_mels, fmin=fmin, fmax=fmax, alpha=alpha)
    
    @torch.no_grad()
    def predict_midi(
        self,
        samples,
        sr=SR,
        threshold=THRESHOLD,
        threshold_pred=THRESHOLD_PRED,
        alpha_audio=ALPHA_AUDIO,
        hop_sec=HOP_SEC,
        decay_base=0.5,
        decay_growth=2.0,
        max_duration=10.0,
        min_onset_interval=0.2,
        ticks_per_beat=480,
        tempo=500_000
    ):
        self.audio_model.eval()
        self.prior_model.eval()

        mel = self.audio_model.wav_to_mel(samples, sr=sr, hop_sec=hop_sec)
        mel = mel.transpose(0, 1).unsqueeze(0).to(self.device)  # (1, T, F)

        fused_onset, fused_sustain = self.forward(mel, threshold_pred=threshold_pred, alpha_audio=alpha_audio)

        onset_probs = fused_onset.cpu().numpy()
        sustain_probs = fused_sustain.cpu().numpy()

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
    
    def wav_to_midi_file(self, input_wav, output_midi, sr=SR, threshold=THRESHOLD, threshold_pred=THRESHOLD_PRED, alpha_audio=ALPHA_AUDIO):

        samples, sr = self.load_wav(input_wav, sr)

        midi = self.predict_midi(
            samples,
            sr=sr,
            threshold=threshold,
            threshold_pred=threshold_pred,
            alpha_audio=alpha_audio
        )

        output_midi = Path(output_midi)
        output_midi.parent.mkdir(parents=True, exist_ok=True)
        midi.save(str(output_midi))

        return midi
    
    def wav_to_midi_folder(
        self,
        input_folder,
        output_folder,
        sr=SR,
        threshold=THRESHOLD,
        threshold_pred=THRESHOLD_PRED,
        alpha_audio=ALPHA_AUDIO,
        recursive=False
    ):

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

            self.wav_to_midi_file(
                str(wav_path),
                str(midi_path),
                sr=sr,
                threshold=threshold,
                threshold_pred=threshold_pred,
                alpha_audio=alpha_audio
            )

        print("[OK] All files processed.")

    @torch.no_grad()
    def _apply_prior(self, onset_logits, sustain_logits, threshold_pred=THRESHOLD_PRED, alpha_audio=ALPHA_AUDIO):

        onset_probs = torch.sigmoid(onset_logits)
        sustain_probs = torch.sigmoid(sustain_logits)

        if alpha_audio > 0.99:
            return onset_probs, sustain_probs

        T, N = onset_probs.shape

        step = 20
        context = 100

        fused_onset = onset_probs.clone()
        fused_sustain = sustain_probs.clone()

        for t in range(step, T, step):

            context_start = max(0, t - context)
            context_len = t - context_start

            fused_onset_ctx = fused_onset[context_start:t]
            fused_sustain_ctx = fused_sustain[context_start:t]

            onset_bin, sustain_bin = note_matrices_to_binary(
                fused_onset_ctx.cpu().numpy(),
                fused_sustain_ctx.cpu().numpy(),
                hop_sec=HOP_SEC,
                onset_threshold=threshold_pred,
            )

            onset_bin = torch.tensor(onset_bin, device=self.device)
            sustain_bin = torch.tensor(sustain_bin, device=self.device)

            x = torch.stack([onset_bin, sustain_bin], dim=-1)

            if context_len < context:
                pad = torch.zeros(context - context_len, N, 2, device=self.device)
                x = torch.cat([pad, x], dim=0)

            x = x.unsqueeze(0)

            pred_onset_logits, pred_sustain_logits = self.prior_model(x)
            pred_onset_probs = torch.sigmoid(pred_onset_logits[0])
            pred_sustain_probs = torch.sigmoid(pred_sustain_logits[0])

            end = min(t + step, T)
            length = end - t

            pred_onset_probs = pred_onset_probs[:length]
            pred_sustain_probs = pred_sustain_probs[:length]

            audio_slice_onset = fused_onset[t:end]
            audio_slice_sustain = fused_sustain[t:end]

            confidence_onset = torch.abs(audio_slice_onset - 0.5) * 2
            confidence_sustain = torch.abs(audio_slice_sustain - 0.5) * 2

            alpha_dynamic_onset = alpha_audio + (1 - alpha_audio) * confidence_onset
            alpha_dynamic_sustain = alpha_audio + (1 - alpha_audio) * confidence_sustain

            fused_onset[t:end] = (
                alpha_dynamic_onset * audio_slice_onset
                + (1 - alpha_dynamic_onset) * pred_onset_probs
            )

            fused_sustain[t:end] = (
                alpha_dynamic_sustain * audio_slice_sustain
                + (1 - alpha_dynamic_sustain) * pred_sustain_probs
            )

        fused_onset = fused_onset.clamp(0.0, 1.0)
        fused_sustain = fused_sustain.clamp(0.0, 1.0)

        return fused_onset, fused_sustain
        
def note_matrices_to_binary(
    onset_matrix,
    sustain_matrix,
    hop_sec=HOP_SEC,
    onset_threshold=THRESHOLD_PRED,
    decay_base=0.4,
    max_duration=10.0,
    decay_growth=2.0,
    min_onset_interval=0.2
):

    T, N = sustain_matrix.shape

    onset_bin = np.zeros((T, N), dtype=np.float32)
    sustain_bin = np.zeros((T, N), dtype=np.float32)

    active_notes = {}
    last_onset_time = {}

    for t in range(T):
        time = t * hop_sec

        for p in range(N):

            onset_val = onset_matrix[t, p]
            sustain_val = sustain_matrix[t, p]

            # -------------------------
            # ONSET
            # -------------------------
            if onset_val >= onset_threshold:

                if p in last_onset_time:
                    if time - last_onset_time[p] < min_onset_interval:
                        continue

                last_onset_time[p] = time

                if p in active_notes:
                    del active_notes[p]

                active_notes[p] = {
                    "start_frame": t,
                    "power": 1.0
                }

                onset_bin[t, p] = 1.0
                sustain_bin[t, p] = 1.0
                continue

            # -------------------------
            # SUSTAIN / DECAY
            # -------------------------
            if p in active_notes:

                start_frame = active_notes[p]["start_frame"]
                start_time = start_frame * hop_sec
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
                    del active_notes[p]
                else:
                    sustain_bin[t, p] = 1.0

    return onset_bin, sustain_bin