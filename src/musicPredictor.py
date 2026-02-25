import torch
import torch.nn as nn

NB_NOTES = 88

class MusicPredictor(nn.Module):
    def __init__(
        self,
        weights_path="weights/musicPredictor_weights.pth",
        device=None,
        n_notes=NB_NOTES,
        input_steps=100,
        output_steps=20,
        hidden_size=394,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.n_notes = n_notes
        self.input_steps = input_steps
        self.output_steps = output_steps

        self.frame_dim = n_notes * 2

        self.input_proj = nn.Sequential(
            nn.Linear(self.frame_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        # Encoder GRU
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Decoder GRU
        self.decoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.output_proj = nn.Linear(hidden_size, self.frame_dim)

        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location='cpu')
            self.load_state_dict(state_dict)

    def forward(self, x):
        B, T, N, C = x.shape

        # (B, 100, 176)
        x = x.reshape(B, T, N * C)

        x = self.input_proj(x)

        # Encoder
        _, hidden = self.encoder(x)

        # Decoder
        decoder_input = x[:, -1:, :]

        outputs = []
        hidden_dec = hidden

        for _ in range(self.output_steps):

            out, hidden_dec = self.decoder(decoder_input, hidden_dec)
            # out = (B,1,hidden_size)

            outputs.append(out)

            decoder_input = out

        # (B,20,hidden_size)
        out = torch.cat(outputs, dim=1)

        out = self.output_proj(out)  # (B,20,176)

        out = out.view(B, self.output_steps, self.n_notes, 2)

        onset_logits = out[..., 0]
        sustain_logits = out[..., 1]

        return onset_logits, sustain_logits