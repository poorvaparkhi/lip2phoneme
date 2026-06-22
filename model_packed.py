import torch
import torch.nn as nn
import torch.nn.functional as f


class TinyLip2PhonemeCTC(nn.Module):

    def __init__(self, num_classes, hidden_dim=128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.rnn = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x, input_lens):
        """
        x: [B, T, 1, 96, 96]
        input_lens: [B]
        """

        B, T, C, H, W = x.shape

        # Step 1: run CNN on each frame independently
        x = x.reshape(B * T, C, H, W)      # [B*T, 1, 96, 96]
        x = self.cnn(x)                    # [B*T, 64, 1, 1]
        x = x.reshape(B, T, 64)            # [B, T, 64]

        # Step 2: pack the sequence before LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            input_lens.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # Step 3: 2-layer BiLSTM processes only real frames
        packed_out, _ = self.rnn(packed)

        # Step 4: convert packed output back to normal padded tensor
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=T,
        )                                  # [B, T, 2 * hidden_dim]

        # Step 5: classify every timestep
        logits = self.classifier(x)        # [B, T, num_classes]

        log_probs = f.log_softmax(logits, dim=-1)

        # CTC expects [T, B, C]
        log_probs = log_probs.permute(1, 0, 2).contiguous()

        return log_probs, input_lens