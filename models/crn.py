"""
CRN-Lite: convolutional encoder -> GRU bottleneck -> convolutional
decoder with skip connections. Predicts an IRM-style mask.

Spec: docs/advanced_model.md (sections 4.1-4.3).

Input : normalized log-magnitude  [B, T, 257]
Output: mask in (0, 1)            [B, T, 257]

Same tensor shapes as the GRU baseline, so the training and
evaluation scripts only need to swap the model class.
"""

import torch
import torch.nn as nn

N_FREQ = 257

# Frequency size after each encoder layer: 257 -> 129 -> 65 -> 33 -> 17
ENC_CHANNELS = [16, 32, 64, 64]


def enc_block(c_in, c_out):
    """Conv that halves frequency, keeps time."""
    return nn.Sequential(
        nn.Conv2d(
            c_in, c_out,
            kernel_size=(3, 3),
            stride=(2, 1),
            padding=(1, 1),
        ),
        nn.BatchNorm2d(c_out),
        nn.ELU(),
    )


def dec_block(c_in, c_out, last=False):
    """
    Transposed conv that doubles frequency (F -> 2F - 1), keeps time.

    Note: design doc lists output_padding=(1, 0). With k=3, s=2, p=1
    that gives 34 instead of 33, so output_padding must be (0, 0)
    for an exact mirror of the encoder (17->33->65->129->257).
    """
    conv = nn.ConvTranspose2d(
        c_in, c_out,
        kernel_size=(3, 3),
        stride=(2, 1),
        padding=(1, 1),
        output_padding=(0, 0),
    )
    if last:
        return nn.Sequential(conv, nn.Sigmoid())
    return nn.Sequential(conv, nn.BatchNorm2d(c_out), nn.ELU())


class CRNDenoiser(nn.Module):
    def __init__(self, n_freq=N_FREQ, hidden_size=256):
        super().__init__()
        c1, c2, c3, c4 = ENC_CHANNELS

        # Encoder
        self.e1 = enc_block(1, c1)
        self.e2 = enc_block(c1, c2)
        self.e3 = enc_block(c2, c3)
        self.e4 = enc_block(c3, c4)

        # Bottleneck: freq after 4 halvings
        f = n_freq
        for _ in range(4):
            f = (f - 1) // 2 + 1
        self.bott_freq = f              # 17
        self.bott_ch = c4               # 64
        flat = c4 * f                   # 1088

        self.gru = nn.GRU(
            flat, hidden_size,
            num_layers=1, batch_first=True,
        )
        self.proj = nn.Linear(hidden_size, flat)

        # Decoder: input = upstream + skip (concat on channels)
        self.d4 = dec_block(c4 + c4, c3)   # 128 -> 64
        self.d3 = dec_block(c3 + c3, c2)   # 128 -> 32
        self.d2 = dec_block(c2 + c2, c1)   #  64 -> 16
        self.d1 = dec_block(c1 + c1, 1, last=True)  # 32 -> 1

    def forward(self, x):
        # x: [B, T, F] -> [B, 1, F, T]
        b, t, _ = x.shape
        x = x.permute(0, 2, 1).unsqueeze(1)

        s1 = self.e1(x)    # [B, 16, 129, T]
        s2 = self.e2(s1)   # [B, 32,  65, T]
        s3 = self.e3(s2)   # [B, 64,  33, T]
        s4 = self.e4(s3)   # [B, 64,  17, T]

        # [B, C, F, T] -> [B, T, C*F]
        z = s4.permute(0, 3, 1, 2).reshape(b, t, -1)
        z, _ = self.gru(z)
        z = self.proj(z)
        # back to [B, C, F, T]
        z = z.reshape(b, t, self.bott_ch, self.bott_freq)
        z = z.permute(0, 2, 3, 1)

        y = self.d4(torch.cat([z, s4], dim=1))  # 33
        y = self.d3(torch.cat([y, s3], dim=1))  # 65
        y = self.d2(torch.cat([y, s2], dim=1))  # 129
        y = self.d1(torch.cat([y, s1], dim=1))  # 257

        # [B, 1, F, T] -> [B, T, F]
        return y.squeeze(1).permute(0, 2, 1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
