import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepLocModel(nn.Module):
    def __init__(self, max_len=1000, vocab_size=20, embed_dim=32):
        super(DeepLocModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # CNN block
        kernel_sizes = [1, 3, 5, 9, 15, 21]
        num_filters = 20  # per kernel size => total 120
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k, padding=k // 2) for k in kernel_sizes]
        )

        # Second convolutional layer (after concatenation of 120 features)
        self.conv2 = nn.Conv1d(120, 128, kernel_size=3, padding=1)

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # Attention Decoder
        self.decoder_lstm = nn.LSTMCell(input_size=1024, hidden_size=512)

        self.attn_v = nn.Parameter(torch.randn(256))
        self.W_e = nn.Linear(512, 256)
        self.W_d = nn.Linear(512, 256)

        # Final layers
        self.fc = nn.Linear(512, 512)
        self.output = nn.Linear(512, 8)  # 8 locations

        self.num_decoding_steps = 10

        # Learned initial context vector
        self.c0 = nn.Parameter(torch.randn(512))

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embed(x)  # (batch_size, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embed_dim, seq_len)

        # CNN: concat outputs of different filter sizes
        cnn_outs = [
            F.relu(conv(x)) for conv in self.convs
        ]  # List of (B, num_filters, seq_len)
        cnn_cat = torch.cat(cnn_outs, dim=1)  # (B, 120, seq_len)
        cnn_out = F.relu(self.conv2(cnn_cat))  # (B, 128, seq_len)
        cnn_out = cnn_out.permute(0, 2, 1)  # (B, seq_len, 128)

        # BiLSTM
        lstm_out, _ = self.bilstm(cnn_out)  # (B, seq_len, 512)

        # Attention Decoder
        batch_size, seq_len, hidden_dim = lstm_out.size()
        decoder_hidden = torch.zeros(batch_size, 512, device=x.device)
        decoder_cell = torch.zeros(batch_size, 512, device=x.device)
        context = self.c0.unsqueeze(0).expand(batch_size, -1)  # (B, 256)

        for _ in range(self.num_decoding_steps):
            decoder_input = torch.cat([context, decoder_hidden], dim=1)  # (B, 1024)
            decoder_hidden, decoder_cell = self.decoder_lstm(
                decoder_input, (decoder_hidden, decoder_cell)
            )

            # Attention mechanism
            e = torch.tanh(
                self.W_e(lstm_out) + self.W_d(decoder_hidden).unsqueeze(1)
            )  # (B, seq_len, 256)
            a = torch.matmul(e, self.attn_v)  # (B, seq_len)
            attn_weights = F.softmax(a, dim=1)  # (B, seq_len)

            context = torch.sum(attn_weights.unsqueeze(2) * lstm_out, dim=1)  # (B, 512)

        # Final dense layers
        x = F.relu(self.fc(context))  # (B, 512)
        out = torch.sigmoid(self.output(x))  # (B, 8) for multi-label
        return out
