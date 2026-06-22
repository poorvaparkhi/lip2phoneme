import torch
import torch.nn as nn
import torch.nn.functional as f

class TinyLip2PhonemeCTC(nn.Module):

    def  __init__(self, num_classes, hidden_dim=128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))
        
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True,
            bidirectional=True,) 
        
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, input_lens):
            """
            x: [B, T, 1, 96, 96]
            input_lens: [B]
            """

            B, T, C, H, W = x.shape

            x = x.reshape(B * T, C, H, W)
            x = self.cnn(x)
            x = x.view(B, T, 64)

            x, _ = self.rnn(x)

            logits = self.classifier(x)

            log_probs = f.log_softmax(logits, dim=-1)
            log_probs = log_probs.permute(1, 0, 2)

            return log_probs, input_lens