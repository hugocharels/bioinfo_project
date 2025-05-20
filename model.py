import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# DeepLoc Model
class DeepLocModel(nn.Module):
    def __init__(self, input_dim=20, max_seq_len=1000):
        super(DeepLocModel, self).__init__()
        self.max_seq_len = max_seq_len
        kernel_sizes = [1, 3, 5, 9, 15, 21]
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=20, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.second_conv = nn.Conv1d(in_channels=120, out_channels=128, kernel_size=3, padding=1)
        self.bilstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True, bidirectional=True)
        self.decoder_lstm = nn.LSTMCell(512, 512)
        self.attn_fc = nn.Linear(512 + 512, 256)
        self.attn_score = nn.Linear(256, 1)
        self.decoder_steps = 10
        self.fc = nn.Linear(512, 512)
        self.output_localization = nn.Linear(512, 10)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv_outs = [F.relu(conv(x)) for conv in self.conv_layers]
        x = torch.cat(conv_outs, dim=1)
        x = self.second_conv(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        batch_size, seq_len, _ = lstm_out.size()
        decoder_hidden = torch.zeros(batch_size, 512).to(x.device)
        decoder_cell = torch.zeros(batch_size, 512).to(x.device)
        context = torch.zeros(batch_size, 512).to(x.device)
        for _ in range(self.decoder_steps):
            decoder_hidden, decoder_cell = self.decoder_lstm(context, (decoder_hidden, decoder_cell))
            expanded_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
            attn_input = torch.cat([lstm_out, expanded_hidden], dim=2)
            attn_weights = self.attn_score(torch.tanh(self.attn_fc(attn_input))).squeeze(-1)
            attn_weights = F.softmax(attn_weights, dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        x = F.relu(self.fc(context))
        y_localization = self.output_localization(x)
        return y_localization
