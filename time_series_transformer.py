import random
from transformer import *
import torch
import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    """ Transformer network architecture for time series """

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
        # a linear layer to convert the vector of variables input into standard model_size (which could be thought as
        # hidden/latent state variables)
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
        # decoder positional encoding layer is probably not necessary. Probably not important to the model performance
        self.target_pos = PositionalEncoding(embedding_size=model_size,
                                             dropout=decoder_dropout)
        self.stacked_decoders = nn.ModuleList([Decoder(size=model_size,
                                                       dropout=decoder_dropout,
                                                       num_heads=decoder_num_attn_heads) for _ in range(num_decoder)])
        self.fc = nn.Linear(in_features=model_size,
                            out_features=output_size)

    def forward(self, x_source, target_input_sequence):
        """
        :param x_source: input to the encoder side
        :param target_input_sequence: target input sequence to feed the decoder side
        :return: transformer output
        """
        x = self.source_fc(x_source)
        x = self.source_pos(x)
        for i in range(len(self.stacked_encoders)):
            x = self.stacked_encoders[i](x)

        y = target_input_sequence
        y = self.target_fc(y)
        y = self.target_pos(y)
        for i in range(len(self.stacked_decoders)):
            y = self.stacked_decoders[i](y, x)
        y = self.fc(y)

        return y

    def forecast(self, x_source, target_seed):
        """
        Forecast future time series with length of forward_window
        :param x_source: input sequence
        :param target_seed: the seed for output. Usually we set as the last element of x_source if tgt_cols is
                        the same as src_cols
        :return: forecasted time series with length of forward_window
        """
        self.eval()

        x = self.source_fc(x_source)
        x = self.source_pos(x)
        for i in range(len(self.stacked_encoders)):
            x = self.stacked_encoders[i](x)

        y = target_seed
        _, y_seq_len, _ = y.size()
        for step in range(self.forward_window):
            output = self.target_fc(y)
            output = self.target_pos(output)
            for i in range(len(self.stacked_decoders)):
                output = self.stacked_decoders[i](output, x)
            output = self.fc(output)

            # use the last sequence and append to the end of input
            next_element = output[:, -1, :].unsqueeze(1)
            y = torch.cat((y.detach(), next_element.detach()), dim=1)

        # output2 would be what decoder first predicted at each step.  output would be what is predicted at last step.
        # The first predicted element are decoded again so they would be different than what is predicted at the last
        # step. output2 seems to be more logical choice based on the autoregressive nature of decoding. But why not use
        # output which has more decoder iterations for the first few elements
        output2 = y[:, y_seq_len:, :]

        return output2
        # return output


class TimeSeriesGPT(nn.Module):
    """ Decoder only network architecture for time series """

    def __init__(self, input_size, model_size, output_size, num_decoder, decoder_dropout,
                 decoder_num_attn_heads):
        """

        :param model_size: model size is basically the embedding size which is similar to the hidden state
        :param num_decoder: number of decoders in the stack
        :param decoder_dropout: dropout for decoder side
        :param decoder_num_attn_heads: number of attention heads for decoder multi head attention
        """
        super().__init__()

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

    def forward(self, sequence, teacher_forcing_prob_threshold=0.0):
        """
        :param sequence: target sequence are used as input and target. Input is [:-1] and target is [1:]
        :param teacher_forcing_prob_threshold: probability threshold for teacher forcing. default 0 means no teacher forcing
        :return:
        """
        # start with the first element in the time series
        y = sequence[:, 0, :].unsqueeze(1)
        for step in range(sequence.size()[1] - 1):
            output = self.target_fc(y)
            # if we are generating one item a time, do we really need positional encoding?
            # output = self.target_pos(output)
            for i in range(len(self.stacked_decoders)):
                output = self.stacked_decoders[i](output)
            output = self.fc(output)

            # use the last sequence as prediction for next element in time series
            pred = output[:, -1, :].unsqueeze(1)
            if self.teacher_forcing(teacher_forcing_prob_threshold):
                # pick from the target. don't forget to use detach()
                y = torch.cat((y.detach(), sequence[:, step + 1, :].unsqueeze(1).detach()), dim=1)
            else:
                y = torch.cat((y.detach(), pred.detach()), dim=1)

        # return a sequence that is shifted to the right
        return output

    def teacher_forcing(self, prob_threshold):
        """ determine if we should teacher force based on the threshold """
        return random.random() < prob_threshold


class TimeSeriesBERT(nn.Module):
    def __init__(self):
        super().__init__()