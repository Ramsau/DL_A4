import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, bidirectional=False):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.bidirectional = bidirectional
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(rnn_output_dim, 1)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        lengths_cpu = lengths.cpu().to(torch.int64)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.rnn(packed)

        if self.bidirectional:
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            hidden = h_n[-1]
        output = self.fc(hidden)
        return output.squeeze(1)
