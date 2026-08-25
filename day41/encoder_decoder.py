from abc import ABC, abstractmethod

import torch
from torch import nn


class Encoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass


class Decoder(nn.Module, ABC):
    @abstractmethod
    def init_state(self, encoded):
        pass

    @abstractmethod
    def forward(self, x, state):
        pass


class GRUEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        return self.rnn(self.embedding(x))


class GRUDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def init_state(self, encoded):
        _, state = encoded
        return state

    def forward(self, x, state):
        output, state = self.rnn(self.embedding(x), state)
        return self.output(output), state


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input, decoder_input):
        encoded = self.encoder(encoder_input)
        state = self.decoder.init_state(encoded)
        return self.decoder(decoder_input, state)


if __name__ == "__main__":
    model = EncoderDecoder(GRUEncoder(20, 8, 16), GRUDecoder(25, 8, 16))
    source = torch.randint(0, 20, (4, 7))
    target = torch.randint(0, 25, (4, 6))
    output, state = model(source, target)
    print(output.shape, state.shape)
