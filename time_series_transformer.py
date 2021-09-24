import random
from transformer import *
import math
import torch
import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    """ Transformer network architecture for both seq2seq and time series """
    def __init__(self, input_size, model_size, output_size, num_encoder, num_decoder, encoder_dropout, decoder_dropout,
                 encoder_num_attn_heads, decoder_num_attn_heads, forward_window):
        """

        :param model_size: model size is basically the embedding size which is similar to the hidden state
        :param num_encoder: number of encoders in the stack
        :param num_decoder: number of decoders in the stack
        :param encoder_dropout: dropout for encoder side
        :param decoder_dropout: dropout for decoder side
        :param encoder_num_attn_heads: number of attention heads for encoder multi head attention
        :param decoder_num_attn_heads: number of attention heads for decoder multi head attention
        :param forward_window: projection period
        """
        super().__init__()
        self.forward_window = forward_window

        # encoder side
        self.source_fc = nn.Linear(in_features=input_size,
                                   out_features=model_size)
        self.source_pos = PositionalEncoding(embedding_size=model_size,
                                             dropout=encoder_dropout)
        self.stacked_encoders = nn.ModuleList([Encoder(size=model_size,
                                                       dropout=encoder_dropout,
                                                       num_heads=encoder_num_attn_heads) for _ in range(num_encoder)])

        # decoder side
        self.target_fc = nn.Linear(in_features=output_size,
                                   out_features=model_size)
        self.target_pos = PositionalEncoding(embedding_size=model_size,
                                             dropout=decoder_dropout)
        self.stacked_decoders = nn.ModuleList([Decoder(size=model_size,
                                                       dropout=decoder_dropout,
                                                       num_heads=decoder_num_attn_heads) for _ in range(num_decoder)])
        self.fc = nn.Linear(in_features=model_size,
                            out_features=output_size)

    def forward(self, x_source, target_input_sequence, target_sequence, teacher_forcing_prob_threshold=0.0):
        """

        :param x_source:
        :param target_input_sequence: target input sequence to feed the decoder side
        :param target_sequence: target
        :param teacher_forcing_prob_threshold: probability threshold for teacher forcing. default 0 means no teacher forcing
        :return:
        """
        x = self.source_fc(x_source)
        x = self.source_pos(x)
        for i in range(len(self.stacked_encoders)):
            x = self.stacked_encoders[i](x)

        y = target_input_sequence
        for step in range(self.forward_window):
            output = self.target_fc(y)
            output = self.target_pos(output)
            for i in range(len(self.stacked_decoders)):
                output = self.stacked_decoders[i](output, x)
            output = self.fc(output)

            # use the last sequence
            pred = output[:, -1, :].unsqueeze(1)
            if self.teacher_forcing(teacher_forcing_prob_threshold) and (step < self.forward_window - 1):
                # pick from the target
                y = torch.cat((y[:, 1:, :], target_sequence[:, step, :].unsqueeze(1)), dim=1)
            else:
                y = torch.cat((y[:, 1:, :], pred), dim=1)

        return y

    def teacher_forcing(self, prob_threshold):
        return random.random() < prob_threshold


class TimeSeriesGPT(nn.Module):
    def __init__(self, input_size, model_size, output_size, num_decoder, decoder_dropout,
                 decoder_num_attn_heads, forward_window):
        """

        :param model_size: model size is basically the embedding size which is similar to the hidden state
        :param num_decoder: number of decoders in the stack
        :param decoder_dropout: dropout for decoder side
        :param decoder_num_attn_heads: number of attention heads for decoder multi head attention
        :param forward_window: projection period
        """
        super().__init__()
        self.forward_window = forward_window

        # decoder side
        self.target_fc = nn.Linear(in_features=input_size,
                                   out_features=model_size)
        self.target_pos = PositionalEncoding(embedding_size=model_size,
                                             dropout=decoder_dropout)
        self.stacked_decoders = nn.ModuleList([Decoder(size=model_size,
                                                       dropout=decoder_dropout,
                                                       num_heads=decoder_num_attn_heads,
                                                       encoder_attention=False) for _ in range(num_decoder)])
        self.fc = nn.Linear(in_features=model_size,
                            out_features=output_size)

    def forward(self, target_input_sequence, target_sequence, teacher_forcing_prob_threshold=0.0):
        """
        :param target_input_sequence: target input sequence to feed the decoder side
        :param target_sequence: target
        :param teacher_forcing_prob_threshold: probability threshold for teacher forcing. default 0 means no teacher forcing
        :return:
        """
        y = target_input_sequence
        for step in range(self.forward_window):
            output = self.target_fc(y)
            output = self.target_pos(output)
            for i in range(len(self.stacked_decoders)):
                output = self.stacked_decoders[i](output)
            output = self.fc(output)

            # use the last sequence
            pred = output[:, -1, :].unsqueeze(1)
            if self.teacher_forcing(teacher_forcing_prob_threshold) and (step < self.forward_window - 1):
                # pick from the target
                y = torch.cat((y[:, 1:, :], target_sequence[:, step, :].unsqueeze(1)), dim=1)
            else:
                y = torch.cat((y[:, 1:, :], pred), dim=1)

        return y[:, -self.forward_window:, :]

    def teacher_forcing(self, prob_threshold):
        return random.random() < prob_threshold


class TimeSeriesBERT(nn.Module):
    def __init__(self):
        super().__init__()