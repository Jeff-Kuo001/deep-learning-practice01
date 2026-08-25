import torch
from torch import nn


class BERTModel(nn.Module):
    def __init__(
        self, vocab_size, hidden_size=32, num_layers=2,
        num_heads=4, ffn_size=64, max_len=128
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.segment_embedding = nn.Embedding(2, hidden_size)
        self.position_embedding = nn.Parameter(torch.randn(1, max_len, hidden_size) * 0.02)
        layer = nn.TransformerEncoderLayer(
            hidden_size, num_heads, ffn_size, batch_first=True,
            activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.mlm = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size),
        )
        self.nsp = nn.Linear(hidden_size, 2)

    def forward(self, tokens, segments, valid_lens, pred_positions=None):
        steps = tokens.size(1)
        x = (
            self.token_embedding(tokens)
            + self.segment_embedding(segments)
            + self.position_embedding[:, :steps]
        )
        positions = torch.arange(steps, device=tokens.device)[None, :]
        padding_mask = positions >= valid_lens[:, None]
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        mlm_output = None
        if pred_positions is not None:
            batch_index = torch.arange(tokens.size(0), device=tokens.device)
            batch_index = batch_index.repeat_interleave(pred_positions.size(1))
            masked = encoded[batch_index, pred_positions.reshape(-1)]
            mlm_output = self.mlm(masked).reshape(
                tokens.size(0), pred_positions.size(1), -1
            )
        nsp_output = self.nsp(encoded[:, 0])
        return encoded, mlm_output, nsp_output


if __name__ == "__main__":
    model = BERTModel(vocab_size=50)
    tokens = torch.randint(0, 50, (2, 10))
    segments = torch.tensor([[0] * 5 + [1] * 5] * 2)
    positions = torch.tensor([[2, 6], [3, 7]])
    encoded, mlm, nsp = model(tokens, segments, torch.tensor([10, 8]), positions)
    print(encoded.shape, mlm.shape, nsp.shape)
