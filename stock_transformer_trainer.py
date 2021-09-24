import torch
import torch.optim as optim
import os
from argparse import Namespace
from trainer import ModelTrainer, sequence_loss, normalize_sizes, calc_accuracy
import pandas as pd
from time_series_transformer import TimeSeriesGPT
from stocks import StockDatasetGPT
import math
from sklearn.preprocessing import MinMaxScaler


class StockTransformerTrainer(ModelTrainer):
    def __init__(self, model, optimizer, scheduler):
        super().__init__(model, optimizer, scheduler)
        self.lossfunc = torch.nn.MSELoss()

    def forward_pass(self, batch_dict):
        return self.model(x_source=batch_dict['x_source'],
                          target_sequence=batch_dict['y_input'])

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
    def __init__(self, model, optimizer, scheduler):
        super().__init__(model, optimizer, scheduler)
        self.lossfunc = torch.nn.MSELoss()

    def forward_pass(self, batch_dict):
        return self.model(target_input_sequence=batch_dict['y_input'],
                          target_sequence=batch_dict['y_target'],
                          teacher_forcing_prob_threshold=self._calc_teacher_forcing_threshold())

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

    def _calc_teacher_forcing_threshold(self):
        k = 20.0
        return k / (k + math.exp(self.train_state['epoch_index'] / k))


if __name__ == "__main__":
    args = Namespace(dataset_csv="sector_etf_logchg.csv",
                     model_state_file="transform_ts_model_pcharm.pth",
                     save_dir="model_storage/stock/",
                     reload_from_files=True,
                     expand_filepaths_to_save_dir=True,
                     cuda=True,
                     seed=1337,
                     learning_rate=5e-4,
                     batch_size=64,
                     num_epochs=300,
                     num_encoder_layer=3,
                     num_decoder_layer=3,
                     num_attn_heads=4,
                     model_size=48,
                     dropout=0.1,
                     early_stopping_criteria=5,
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

    # Set seed for reproducibility
    ModelTrainer.set_seed_everywhere(args.seed, args.cuda)

    # handle dirs
    ModelTrainer.handle_dirs(args.save_dir)

    # create dataset
    src_cols = ['XLF']
    tgt_cols = ['XLF']

    df = pd.read_csv(args.dataset_csv)
    scaler = MinMaxScaler()
    df2 = scaler.fit_transform(df.iloc[:, 1:-1])
    df.iloc[:, 1:-1] = df2

    dataset = StockDatasetGPT(df,
                              lookback_window=20,
                              forward_window=5,
                              src_cols=['XLF'],
                              tgt_cols=['XLF'])

    model = TimeSeriesGPT(input_size=len(src_cols),
                          model_size=args.model_size,
                          output_size=len(tgt_cols),
                          num_decoder=args.num_decoder_layer,
                          decoder_dropout=args.dropout,
                          decoder_num_attn_heads=args.num_attn_heads,
                          forward_window=5)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min', factor=0.5,
                                                     patience=1)

    trainer = StockGPTTrainer(model, optimizer, scheduler)
    train_state = trainer.train(dataset=dataset, args=args)
    print("training finished: ", train_state)

    # save model
    torch.save(model.state_dict(), args.model_state_file)

