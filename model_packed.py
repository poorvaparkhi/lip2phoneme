import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalShift(nn.Module):
    """
    Temporal Shift Module for tensors shaped [B, T, C, H, W].

    Bidirectional/offline TSM:
        - 1/fold_div channels take information from the next frame
        - 1/fold_div channels take information from the previous frame
        - remaining channels stay at the current frame

    This adds temporal mixing without extra learnable parameters.
    """

    def __init__(self, fold_div: int = 8):
        super().__init__()
        if fold_div < 2:
            raise ValueError(f"fold_div must be >= 2, got {fold_div}")
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          [B, T, C, H, W]
            valid_mask: [B, T], True for real frames, False for padding
        """
        if x.ndim != 5:
            raise ValueError(f"Expected x [B,T,C,H,W], got {tuple(x.shape)}")

        B, T, C, H, W = x.shape
        fold = C // self.fold_div

        if fold == 0 or T == 1:
            return x * valid_mask[:, :, None, None, None].to(x.dtype)

        out = torch.zeros_like(x)

        # Channels [0:fold] receive information from the next frame.
        # At timestep t, this stores x[t+1]. Useful for offline/bidirectional models.
        out[:, :-1, :fold] = x[:, 1:, :fold]

        # Channels [fold:2*fold] receive information from the previous frame.
        # At timestep t, this stores x[t-1].
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]

        # Remaining channels stay at the current timestep.
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        # Keep padded timesteps exactly zero.
        out = out * valid_mask[:, :, None, None, None].to(out.dtype)
        return out


class TinyLip2PhonemeCTC(nn.Module):
    """
    Visual front end with TSM-based local temporal context:

        padded video frames
            -> CNN block 1 -> TSM
            -> CNN block 2 -> TSM
            -> CNN block 3 -> TSM + 3x3 adaptive pooling
            -> per-frame projection
            -> packed 2-layer BiLSTM
            -> CTC logits

    TSM keeps the sequence length unchanged, so CTC output_lens == input_lens.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 128,
        pooled_size: int = 3,
        temporal_dim: int = 192,
        temporal_dropout: float = 0.10,
        rnn_dropout: float = 0.20,
        tsm_fold_div: int = 8,
    ):
        super().__init__()

        if pooled_size < 1:
            raise ValueError(f"pooled_size must be >= 1, got {pooled_size}")

        self.pooled_size = pooled_size
        self.temporal_dim = temporal_dim
        self.tsm = TemporalShift(fold_div=tsm_fold_div)

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((pooled_size, pooled_size)),
        )

        cnn_feature_dim = 64 * pooled_size * pooled_size

        self.frame_proj = nn.Sequential(
            nn.Linear(cnn_feature_dim, temporal_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(temporal_dropout),
        )

        self.rnn = nn.LSTM(
            input_size=temporal_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout,
        )

        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def _run_cnn_block_with_tsm(
        self,
        x: torch.Tensor,
        block: nn.Module,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          [B, T, C, H, W]
            block:      2D CNN block
            valid_mask: [B, T]

        Returns:
            [B, T, C_out, H_out, W_out]
        """
        B, T, C, H, W = x.shape

        # 2D CNN block sees frames as independent images.
        x = x.reshape(B * T, C, H, W)
        x = block(x)

        C2, H2, W2 = x.shape[1], x.shape[2], x.shape[3]
        x = x.reshape(B, T, C2, H2, W2)

        # TSM mixes neighboring timesteps after this CNN block.
        x = self.tsm(x, valid_mask)
        return x

    def forward(self, x: torch.Tensor, input_lens: torch.Tensor):
        """
        Args:
            x:          [B, T, 1, 96, 96], padded on the right in time.
            input_lens: [B], number of real frames in each utterance.

        Returns:
            log_probs:   [T, B, num_classes] for nn.CTCLoss.
            output_lens: [B], identical to input_lens.
        """
        if x.ndim != 5:
            raise ValueError(
                f"Expected x with 5 dimensions [B,T,C,H,W], got {tuple(x.shape)}"
            )

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

        time_index = torch.arange(T, device=x.device).unsqueeze(0)
        valid_mask = time_index < input_lens.unsqueeze(1)  # [B, T]

        # Zero padded frames before CNN. Length-bucketed batches reduce how many
        # padded frames enter BatchNorm.
        x = x * valid_mask[:, :, None, None, None].to(x.dtype)

        x = self._run_cnn_block_with_tsm(x, self.block1, valid_mask)
        x = self._run_cnn_block_with_tsm(x, self.block2, valid_mask)
        x = self._run_cnn_block_with_tsm(x, self.block3, valid_mask)

        # [B, T, 64, P, P] -> [B, T, 64*P*P]
        seq = x.flatten(start_dim=2)

        # [B, T, 64*P*P] -> [B, T, temporal_dim]
        seq = self.frame_proj(seq)
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
        )

        logits = self.classifier(seq)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs.permute(1, 0, 2).contiguous(), input_lens
