import math

import torch


class GeLU(torch.nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )


class LayerNorm(torch.nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = torch.nn.Parameter(torch.ones(features))
        self.b_2 = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(x.dim() - 1, keepdim=True)
        std = x.std(x.dim() - 1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_in, d_h, d_out, dropout=0.1):
        super().__init__()
        self.w_1 = torch.nn.Linear(d_in, d_h)
        self.w_2 = torch.nn.Linear(d_h, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = GeLU()
        self.ln = LayerNorm(d_out)

    def forward(self, x):
        return self.ln(self.w_2(self.dropout(self.activation(self.w_1(x)))))


class ImageCLIP(torch.nn.Module):
    def __init__(self, model, top_feedforward=False, d_h=None, d_out=None):
        super().__init__()
        self.model = model
        self.embedding_dim = model.visual.output_dim
        self.top_feedforward = top_feedforward
        if self.top_feedforward:
            d_h = self.embedding_dim // 2 * 3 if d_h is None else d_h
            d_out = self.embedding_dim // 2 if d_out is None else d_out
            self.ff = PositionwiseFeedForward(
                d_in=self.embedding_dim,
                d_h=d_h,
                d_out=d_out,
            )

    def forward(self, image):
        out = self.model.encode_image(image)
        return self.ff(out) if self.top_feedforward else out


class TextCLIP(torch.nn.Module):
    def __init__(self, model, top_feedforward=False, d_h=None, d_out=None):
        super().__init__()
        self.model = model
        self.embedding_dim = model.transformer.width
        self.top_feedforward = top_feedforward
        if self.top_feedforward:
            d_h = self.embedding_dim // 2 * 3 if d_h is None else d_h
            d_out = self.embedding_dim // 2 if d_out is None else d_out
            self.ff = PositionwiseFeedForward(
                d_in=self.embedding_dim,
                d_h=d_h,
                d_out=d_out,
            )

    def forward(self, text):
        out = self.model.encode_text(text)
        return self.ff(out) if self.top_feedforward else out
