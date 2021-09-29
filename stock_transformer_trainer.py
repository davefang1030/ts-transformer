import torch
import torch.optim as optim
import os
from argparse import Namespace
from trainer import ModelTrainer
import pandas as pd
from time_series_transformer import TimeSeriesGPT, TimeSeriesTransformer
from stocks import StockDatasetGPT, StockDataset
import math


class StockTransformerTrainer(ModelTrainer):
    """ transformer time series trainer """
    def __init__(self, model, optimizer, scheduler):
        super().__init__(model, optimizer, scheduler)
        self.lossfunc = torch.nn.MSELoss()

    def forward_pass(self, batch_dict):
        return self.model(x_source=batch_dict['x_source'],
                          target_input_sequence=batch_dict['y_input'])

    def calculate_loss(self, y_pred, batch_dict, mask_index):
        return self.lossfunc(y_pred, batch_dict['y_target'])

    def compute_accuracy(self, y_pred, batch_dict, mask_index):
        """
        For MSELoss, there is no accuracy to calculate and always return 0
        :param y_pred:
        :param batch_dict:
        :param mask_index:
        :return:
        """
        return 0


class StockGPTTrainer(ModelTrainer):
    """ Decoder only time series trainer """
    def __init__(self, model, optimizer, scheduler):
        super().__init__(model, optimizer, scheduler)
        self.lossfunc = torch.nn.MSELoss()

    def forward_pass(self, batch_dict):
        return self.model(sequence=batch_dict['y_input'],
                          teacher_forcing_prob_threshold=self._calc_teacher_forcing_threshold())

    def calculate_loss(self, y_pred, batch_dict, mask_index):
        target = batch_dict['y_input']
        target = target[:, 1:, :]
        return self.lossfunc(y_pred, target)

    def compute_accuracy(self, y_pred, batch_dict, mask_index):
        """
        For MSELoss, there is no accuracy to calculate and always return 0
        :param y_pred:
        :param batch_dict:
        :param mask_index:
        :return:
        """
        return 0

    def _calc_teacher_forcing_threshold(self):
        k = 15.0
        return k / (k + math.exp(self.train_state['epoch_index'] / k))


if __name__ == "__main__":
    args = Namespace(dataset_csv="sector_etf.csv",
                     model_state_file="transform_ts_model.pth",
                     save_dir="model_storage/stock/",
                     reload_from_files=True,
                     expand_filepaths_to_save_dir=True,
                     cuda=True,
                     seed=1337,
                     learning_rate=1e-4,
                     batch_size=64,
                     num_epochs=20,
                     num_encoder_layer=12,
                     num_decoder_layer=4,
                     num_attn_heads=4,
                     model_size=64,
                     dropout=0.2,
                     early_stopping_criteria=10,
                     forward_window=5,
                     lookback_window=50,
                     sequence_length=10,
                     catch_keyboard_interrupt=True)

    if args.expand_filepaths_to_save_dir:
        args.model_state_file = os.path.join(args.save_dir,
                                             args.model_state_file)
        print("Expanded filepaths: ")
        print("\t{}".format(args.model_state_file))

    # Check CUDA
    if not torch.cuda.is_available():
        args.cuda = False
    args.device = torch.device("cuda" if args.cuda else "cpu")
    print("Using CUDA: {}".format(args.cuda))

    # create dataset
    src_cols = ['XLF', 'XLE', 'XLK']
    tgt_cols = ['XLF', 'XLE', 'XLK']

    df = pd.read_csv(args.dataset_csv)
    dataset = StockDataset(df,
                           lookback_window=args.lookback_window,
                           forward_window=args.forward_window,
                           src_cols=src_cols,
                           tgt_cols=tgt_cols)

    model = TimeSeriesTransformer(input_size=len(src_cols),
                                  model_size=args.model_size,
                                  output_size=len(tgt_cols),
                                  num_encoder=args.num_encoder_layer,
                                  num_decoder=args.num_decoder_layer,
                                  encoder_dropout=args.dropout,
                                  decoder_dropout=args.dropout,
                                  encoder_num_attn_heads=args.num_attn_heads,
                                  decoder_num_attn_heads=args.num_attn_heads,
                                  forward_window=args.forward_window)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min', factor=0.5,
                                                     patience=1)

    trainer = StockTransformerTrainer(model, optimizer, scheduler)
    # Set seed for reproducibility
    trainer.set_seed_everywhere(args.seed, args.cuda)
    trainer.handle_dirs(args.save_dir)

    train_state = trainer.train(dataset=dataset, args=args)
    print("\ntraining finished: ", train_state)

    # save model
    # torch.save(model.state_dict(), args.model_state_file)

    import numpy as np
    dataset.set_split('test')
    x_source = np.zeros((len(dataset._target_df), len(src_cols)))
    pred = np.zeros((len(dataset._target_df), len(tgt_cols)))
    # generate input one at a time and no shuffle to keep the time order
    batch_generator = dataset.generate_batches(dataset, batch_size=1, shuffle=False, device=args.device)
    for batch_index, batch_dict in enumerate(batch_generator):
        x_input = batch_dict['x_source']
        y_input = batch_dict['y_input']
        y_target = batch_dict['y_target']
        # forecast
        x_out = model.forecast(x_source=x_input, target_seed=y_input[:, 0, :].unsqueeze(1))
        x_source[batch_index:batch_index+args.lookback_window, :] = x_input.detach().cpu().numpy()
        pred[batch_index+args.lookback_window:batch_index+args.lookback_window+args.forward_window] = \
            x_out.detach().cpu().numpy()

    print("plotting...")
    import matplotlib.pyplot as plt
    for i in range(len(tgt_cols)):
        plt.figure(figsize=(10, 5))
        plt.title(tgt_cols[i])
        plt.plot(x_source[:, i], label='history')
        plt.plot(pred[:, i], label='prediction')
        plt.legend(loc="upper left")
        plt.show()
