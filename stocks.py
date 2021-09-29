from data import PdTorchDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class StockDataset(PdTorchDataset):
    """
    StockDataset represent dataframes with multiple columns of time series data. First column is date and last column
    is split to indicate train/val/test entries. It provides data to be trained by transformer (train on historical
    data with lookback_window length and predict future time series with forward_window length:
        x_source (time series with length of lookback_window), inputs to encoder side
        y_target (time series with length of forward_window), outputs of the decoder
        y_input (time series with length of forward_window), y_target shifted right one spot with last from x_source
    """

    def __init__(self, df, lookback_window, forward_window, src_cols, tgt_cols):
        """
        :param df: dataframe that holds data. All train/val/test data should be continuous wrt dates.
                   we cannot randomize them when we create train/val/test labels.
        :param lookback_window: look back window for source
        :param forward_window: prediction period
        :param src_cols: list of column names for source
        :param tgt_cols: list of column names for target
        """
        # no need for a vectorizer
        super().__init__(df, vectorizer=None)
        self.lookback_window = lookback_window
        self.forward_window = forward_window
        self.src_cols = src_cols
        self.tgt_cols = tgt_cols

        self.train_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.val_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.test_scaler = MinMaxScaler(feature_range=(-1, 1))
        # scale the data and ignore first column (date) and last column (split)
        df1 = self.train_scaler.fit_transform(self.train_df.iloc[:, 1:-1])
        self.train_df.iloc[:, 1:-1] = df1
        df1 = self.val_scaler.fit_transform(self.val_df.iloc[:, 1:-1])
        self.val_df.iloc[:, 1:-1] = df1
        df1 = self.test_scaler.fit_transform(self.test_df.iloc[:, 1:-1])
        self.test_df.iloc[:, 1:-1] = df1

        self.scaler_dict = {'train': self.train_scaler,
                            'val': self.val_scaler,
                            'test': self.test_scaler}

    def __len__(self):
        # override base and return dataframe size minus sequence_length
        return len(self._target_df) - self.lookback_window - self.forward_window

    def __getitem__(self, index):
        """
        starting from index, there are lookback_window obs as input, followed by forward_window obs as target
        :param index:
        :return:
        """
        df1 = self._target_df[index:index+self.lookback_window]
        x_source = df1.loc[:, self.src_cols].to_numpy().astype(np.float32)

        df1 = self._target_df[index+self.lookback_window:index+self.lookback_window+self.forward_window]
        y_target = df1.loc[:, self.tgt_cols].to_numpy().astype(np.float32)

        # decoder input is the target shifted to right with first element from last element of x_source
        df1 = self._target_df[index+self.lookback_window-1:index+self.lookback_window+self.forward_window-1]
        y_input = df1.loc[:, self.tgt_cols].to_numpy().astype(np.float32)

        return {"x_source": x_source,
                "y_input": y_input,
                "y_target": y_target}

    def inverse_transform(self, cols, data):
        """
        inverse transform the data to original space
        :param cols: list of column names
        :param data: ndarray
        :return: data scaled back using scaler inverse_transform()
        """
        num_cols = len(self._target_df.columns)
        num_rows = data.shape[0]
        # set up a matrix without date and split column which has 2 less columns
        data1 = np.zeros((num_rows, num_cols - 2))
        # deduct one from the index as date column is the first one
        index = [self._target_df.columns.get_loc(c) - 1 for c in cols]
        data1[:, index] = data
        inverse = self._target_scaler.inverse_transform(data1)
        return inverse[:, index]

    def set_split(self, split="train"):
        """ override base to provide correct scaler and size """
        super().set_split(split)
        try:
            self._target_size = len(self) - self.lookback_window - self.forward_window
            self._target_scaler = self.scaler_dict[split]
        except:
            # set_split is called before the attributes are created, ignore
            pass


class StockDatasetGPT(PdTorchDataset):
    """
    StockDatasetGPT represent dataframes with multiple columns of time series data. First column is date and last is
    split to indicate train/val/test entries. It provides data to be trained by decoder only (GPT) type of transformer
    (train on historical data with sequence_length length and predict one step future time series:
        y_input (time series with length of sequence_length), the target is shifted right one spot with y_input with
        last element from original series that comes after y_input
    """
    def __init__(self, df, sequence_length, src_cols):
        """
        :param df: dataframe that holds data. All train/val/test data should be continuous wrt dates.
                   we cannot randomize them when we create train/val/test labels.
        :param sequence_length: time series length for training
        :param src_cols: list of column names for source. There is no different tgt_cols as this implementation
                    is meant to test autoregressive feature and teacher forcing
        """
        # no need for a vectorizer
        super().__init__(df, vectorizer=None)
        self.sequence_length = sequence_length
        self.src_cols = src_cols

        self.train_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.val_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.test_scaler = MinMaxScaler(feature_range=(-1, 1))
        # scale the data and ignore first column (date) and last column (split)
        df1 = self.train_scaler.fit_transform(self.train_df.iloc[:, 1:-1])
        self.train_df.iloc[:, 1:-1] = df1
        df1 = self.val_scaler.fit_transform(self.val_df.iloc[:, 1:-1])
        self.val_df.iloc[:, 1:-1] = df1
        df1 = self.test_scaler.fit_transform(self.test_df.iloc[:, 1:-1])
        self.test_df.iloc[:, 1:-1] = df1

        self.scaler_dict = {'train': self.train_scaler,
                            'val': self.val_scaler,
                            'test': self.test_scaler}

    def __len__(self):
        # override base and return dataframe size minus sequence_length
        return len(self._target_df) - self.sequence_length

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        df1 = self._target_df[index:index+self.sequence_length]
        y_input = df1.loc[:, self.src_cols].to_numpy().astype(np.float32)

        return {"y_input": y_input}

    def inverse_transform(self, cols, data):
        """
        inverse transform the data to original space
        :param cols: list of column names
        :param data: ndarray
        :return: data scaled back using scaler inverse_transform()
        """
        num_cols = len(self._target_df.columns)
        num_rows = data.shape[0]
        # set up a matrix without date and split column which has 2 less columns
        data1 = np.zeros((num_rows, num_cols - 2))
        # deduct one from the index as date column is the first one
        index = [self._target_df.columns.get_loc(c) - 1 for c in cols]
        data1[:, index] = data
        inverse = self._target_scaler.inverse_transform(data1)
        return inverse[:, index]

    def set_split(self, split="train"):
        """ override base to provide correct dataframe and size """
        super().set_split(split)
        try:
            self._target_size = len(self) - self.lookback_window - self.forward_window
            self._target_scaler = self.scaler_dict[split]
        except:
            # set_split is called before the attributes are created, ignore
            pass
