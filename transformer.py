import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Transformer(nn.Module):
    """ Transformer network architecture for both seq2seq and time series """
    def __init__(self, model_size, num_encoder, num_decoder, encoder_dropout, decoder_dropout,
                 encoder_num_attn_heads, decoder_num_attn_heads,
                 source_embedding_layer=False, source_emb_size=0,
                 target_embedding_layer=False, target_emb_size=0):
        """

        :param model_size: model size would be the same for embedding size for both encoder and decoder
        :param num_encoder: number of encoders in the stack
        :param num_decoder: number of decoders in the stack
        :param encoder_dropout: dropout for encoder side
        :param decoder_dropout: dropout for decoder side
        :param encoder_num_attn_heads: number of attention heads for encoder multi head attention
        :param decoder_num_attn_heads: number of attention heads for decoder multi head attention
        :param source_embedding_layer: bool, for time series set to false as there is no need for embedding layer
        :param source_emb_size: source vocabulary size
        :param target_embedding_layer: bool, for time series set to false as there is no need for embedding layer
        :param target_emb_size: target vocabulary size
        """
        super().__init__()

        # encoder side
        if source_embedding_layer:
            self.source_emb = nn.Embedding(num_embeddings=source_emb_size,
                                           embedding_dim=model_size)
        else:
            self.source_emb = None

        self.source_pos = PositionalEncoding(embedding_size=model_size,
                                             dropout=encoder_dropout)
        # use ModuleList to ensure modules inside are properly registered and visible
        self.stacked_encoders = nn.ModuleList([Encoder(size=model_size,
                                                       dropout=encoder_dropout,
                                                       num_heads=encoder_num_attn_heads) for _ in range(num_encoder)])

        # decoder side
        if target_embedding_layer:
            self.target_emb = nn.Embedding(num_embeddings=target_emb_size,
                                           embedding_dim=model_size)
        else:
            self.target_emb = None

        self.target_pos = PositionalEncoding(embedding_size=model_size,
                                             dropout=decoder_dropout)
        self.stacked_decoders = nn.ModuleList([Decoder(size=model_size,
                                                       dropout=decoder_dropout,
                                                       num_heads=decoder_num_attn_heads) for _ in range(num_decoder)])

        # if no embedding layer then there is no need for the last full connected layer
        if target_embedding_layer:
            self.fc = nn.Linear(in_features=model_size,
                                out_features=target_emb_size)
        else:
            self.fc = None

    def forward(self, x_source, target_sequence, apply_softmax=False):
        """

        :param x_source:
        :param target_sequence:
        :param apply_softmax:
        :return:
        """
        x = x_source
        if self.source_emb is not None:
            x = self.source_emb(x)
        x = self.source_pos(x)
        for i in range(len(self.stacked_encoders)):
            x = self.stacked_encoders[i](x)

        y = target_sequence
        if self.target_emb is not None:
            y = self.target_emb(y)
        y = self.target_pos(y)
        for i in range(len(self.stacked_decoders)):
            y = self.stacked_decoders[i](y, x)

        if self.fc is not None:
            output = self.fc(y)
            if apply_softmax:
                output = F.softmax(output, dim=1)
        else:
            output = y

        return output


class AddNormLayer(nn.Module):
    """
    Residual & Normalization
    """
    def __init__(self, size, dropout):
        """

        :param size: size of the layer
        :param dropout: dropout probability
        """
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        # Annotated Transformer has implementation below which seems incorrect
        # return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer))


class Encoder(nn.Module):
    def __init__(self, size, dropout, num_heads):
        """
        :param size: model size and embedding layer dimension. Need to be divisible by num_heads
        :param dropout: dropout probability
        :param num_heads: number of heads for multi head attention.
        """
        super().__init__()
        self.attn = MultiHeadAttention(num_heads=num_heads,
                                       model_size=size,
                                       p_dropout=dropout)
        self.add_norm1 = AddNormLayer(size, dropout)
        # not sure if hidden_size is 4 times model_size is more a heurestic
        self.ff = PositionwiseFeedForward(size, 4*size)
        self.add_norm2 = AddNormLayer(size, dropout)

    def forward(self, x):
        encoder_self_attn, _ = self.attn(x, x, x)
        x1 = self.add_norm1(x, encoder_self_attn)
        x2 = self.add_norm2(x1, self.ff(x1))
        return x2


class Decoder(nn.Module):
    def __init__(self, size, dropout, num_heads, encoder_attention=True):
        """

        :param size: model size and embedding layer dimension. Need to be divisible by num_heads
        :param dropout: dropout probability
        :param num_heads: number of heads for multi head attention.
        :param encoder_attention: bool. Does the decoder contain attention block for encoder decoder attention?
        """
        super().__init__()
        self.encoder_attention = encoder_attention

        self.attn_masked = MultiHeadAttention(num_heads=num_heads,
                                              model_size=size,
                                              p_dropout=dropout,
                                              masked=True)
        self.add_norm1 = AddNormLayer(size, dropout)

        if self.encoder_attention:
            self.attn = MultiHeadAttention(num_heads=num_heads,
                                           model_size=size,
                                           p_dropout=dropout,
                                           masked=False)
            self.add_norm2 = AddNormLayer(size, dropout)

        self.ff = PositionwiseFeedForward(size, size*4)
        self.add_norm3 = AddNormLayer(size, dropout)

    def forward(self, y, encoder_output=None):
        decoder_self_attn, _ = self.attn_masked(y, y, y)
        y1 = self.add_norm1(y, decoder_self_attn)
        if self.encoder_attention:
            encoder_decoder_attn, _ = self.attn(y1, encoder_output, encoder_output)
            y2 = self.add_norm2(y1, encoder_decoder_attn)
        else:
            y2 = y1
        y3 = self.add_norm3(y2, self.ff(y2))
        return y3


class ScaleDotProductAttention(nn.Module):
    """
    Dot product attention scaled with square root of the model_size.
    """
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        # linear layers to store trainable weights for query, key and value. add bias here as well
        self.query_fc = nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)
        self.key_fc = nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)
        self.value_fc = nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        q = self.query_fc(query)
        k = self.query_fc(key)
        v = self.query_fc(value)
        d_k = q.size(-1)
        # calculate similarity of query and key
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # set masked value to -1e9 all values in the input of the softmax because exp(-1e9) = 0
            scores = scores.masked_fill(mask == 1, -1e9)
        # softmax for probability which indicates attention level
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        # probability weighted value
        weighted_value = torch.matmul(prob, v)
        return weighted_value, prob


class MultiHeadAttention(nn.Module):
    """
    MultiHead attention mechanism. Holding a list of attention heads and concat attention results as output
    """
    def __init__(self, num_heads, model_size, p_dropout, masked=False):
        """
        :param num_heads: number of heads for multi head attention layer
        :param model_size: model size. Must be divisible by num_heads
        :param p_dropout: dropout probability
        :param masked: bool, set True to prevent peek forward
        """
        super().__init__()
        assert model_size % num_heads == 0
        self.attn_layer_dim = model_size // num_heads
        self.attn_heads = nn.ModuleList([ScaleDotProductAttention(input_dim=model_size,
                                                                  output_dim=self.attn_layer_dim,
                                                                  dropout=p_dropout) for _ in range(num_heads)])
        self.masked = masked

    def forward(self, query, key, value):
        """
        calculate probability weighted values from each head and concat
        :param query: query
        :param key: key
        :param value: value
        :return: probabilities weighted value
        """
        if not self.masked:
            output = [h(query, key, value) for h in self.attn_heads]
        else:
            attn_size = (query.size(0), query.size(1), query.size(1))
            # set element to 1 to mask forward position
            subsequent_mask = torch.from_numpy(np.triu(np.ones(attn_size), k=1).astype('uint8'))
            subsequent_mask = subsequent_mask.to(device=query.device)
            output = [h(query, key, value, subsequent_mask) for h in self.attn_heads]

        # concatenate
        weighted_values, probabilities = zip(*output)
        return torch.cat(weighted_values, dim=2), torch.cat(probabilities, dim=2)


class PositionalEncoding(nn.Module):
    """ Implement positional encoding """
    def __init__(self, embedding_size, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # calculate in log space
        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, embedding_size, 2) *
                                (-math.log(10000.0)) / embedding_size)
        pe[:, 0::2] = torch.sin(position * denominator)
        # we need to handle the case where embedding_size is odd
        if embedding_size % 2 == 1:
            pe[:, 1::2] = torch.cos(position * denominator[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * denominator)

        pe = pe.unsqueeze(0)
        # not a model parameter but part of the model state
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """ Feed forward layer... Two linear layer with ReLu in-between. To add non-linearity? """
    def __init__(self, model_size, hidden_size, dropout=0.1):
        """
        :param model_size: model size
        :param hidden_size: hidden size for first linear output and second linear input
        :param dropout:
        """
        super().__init__()
        self.w_1 = nn.Linear(model_size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, model_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


