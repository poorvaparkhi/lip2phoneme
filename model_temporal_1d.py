import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLip2PhonemeCTC(nn.Module):
    """
    Visual front end with local temporal context:

        valid video frames
            -> CNN with BatchNorm and 3x3 adaptive pooling
            -> per-frame projection
            -> residual temporal Conv1D
            -> packed 2-layer BiLSTM
            -> CTC logits

    The temporal convolution keeps the sequence length unchanged, so the CTC
    output length is still exactly input_lens.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 128,
        pooled_size: int = 3,
        temporal_dim: int = 192,
        temporal_dropout: float = 0.10,
        rnn_dropout: float = 0.20,
    ):
        super().__init__()

        if pooled_size < 1:
            raise ValueError(f"pooled_size must be >= 1, got {pooled_size}")

        self.pooled_size = pooled_size
        self.temporal_dim = temporal_dim

        # CNN runs on individual VALID frames only. This matters with BatchNorm:
        # zero-padded frames should not affect its running batch statistics.
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((pooled_size, pooled_size)),
        )

        cnn_feature_dim = 64 * pooled_size * pooled_size

        # Keep the coarse 3x3 spatial layout, but compress it before Conv1D/LSTM.
        self.frame_proj = nn.Sequential(
            nn.Linear(cnn_feature_dim, temporal_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(temporal_dropout),
        )

        # Two local convolutions give each timestep a 7-frame receptive field
        # (kernel 5 followed by kernel 3), while preserving T.
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                temporal_dim,
                temporal_dim,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(temporal_dropout),
            nn.Conv1d(
                temporal_dim,
                temporal_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )
        self.temporal_norm = nn.LayerNorm(temporal_dim)

        self.rnn = nn.LSTM(
            input_size=temporal_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout,
        )

        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, input_lens: torch.Tensor):
        """
        Args:
            x:          [B, T, 1, 96, 96], padded on the right in time.
            input_lens: [B], number of real frames in each utterance.

        Returns:
            log_probs:   [T, B, num_classes] for nn.CTCLoss.
            output_lens: [B], identical to input_lens because no temporal
                         downsampling is used.
        """
        if x.ndim != 5:
            raise ValueError(f"Expected x with 5 dimensions [B,T,C,H,W], got {tuple(x.shape)}")

        B, T, C, H, W = x.shape
        input_lens = input_lens.to(device=x.device, dtype=torch.long)

        if input_lens.ndim != 1 or input_lens.numel() != B:
            raise ValueError(
                f"input_lens must have shape [{B}], got {tuple(input_lens.shape)}"
            )
        if (input_lens < 1).any() or (input_lens > T).any():
            raise ValueError(
                f"input_lens must be in [1, {T}], got {input_lens.detach().cpu().tolist()}"
            )

        # [B, T], True only for real (not collate-padding) frames.
        time_index = torch.arange(T, device=x.device).unsqueeze(0)
        valid_mask = time_index < input_lens.unsqueeze(1)

        # Avoid feeding zero-padded frames into CNN BatchNorm.
        valid_frames = x[valid_mask]                         # [sum(L_b), C, H, W]
        valid_features = self.cnn(valid_frames)              # [sum(L_b), 64, P, P]
        valid_features = valid_features.flatten(start_dim=1) # [sum(L_b), 64*P*P]
        valid_features = self.frame_proj(valid_features)     # [sum(L_b), temporal_dim]

        # Restore the padded batch layout for Conv1D and packing.
        seq = valid_features.new_zeros(B, T, self.temporal_dim)
        seq[valid_mask] = valid_features                     # [B, T, temporal_dim]

        # Conv1D expects channels before time: [B, D, T].
        temporal_residual = self.temporal_conv(seq.transpose(1, 2)).transpose(1, 2)
        seq = self.temporal_norm(seq + temporal_residual)

        # Make padded positions exactly zero before packing. pack_padded_sequence
        # then removes them completely before the BiLSTM.
        seq = seq * valid_mask.unsqueeze(-1).to(seq.dtype)

        packed = nn.utils.rnn.pack_padded_sequence(
            seq,
            input_lens.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.rnn(packed)

        seq, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=T,
        )                                                    # [B, T, 2*hidden_dim]

        logits = self.classifier(seq)                        # [B, T, num_classes]
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs.permute(1, 0, 2).contiguous(), input_lens