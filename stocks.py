from data import PdTorchDataset
import numpy as np


class StockDataset(PdTorchDataset):
    def __init__(self, df, lookback_window, forward_window, src_cols, tgt_cols):
        """

        :param df: dataframe that holds data
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

    def __getitem__(self, index):
        """
        starting from index, there are lookback_window obs as input, followed by forward_window obs as target
        :param index:
        :return:
        """
        row_idx = self._target_df.index[index]
        idx = list(self.df.index).index(row_idx)

        df1 = self.df[idx:idx+self.lookback_window]
        x_source = df1.loc[:, self.src_cols].to_numpy().astype(np.float32)

        df1 = self.df[idx+self.lookback_window:idx+self.lookback_window+self.forward_window]
        y_target = df1.loc[:, self.tgt_cols].to_numpy().astype(np.float32)

        df1 = self.df[idx+self.lookback_window-self.forward_window:idx+self.lookback_window]
        y_input = df1.loc[:, self.tgt_cols].to_numpy().astype(np.float32)

        return {"x_source": x_source,
                "y_input": y_input,
                "y_target": y_target}


class StockDatasetGPT(PdTorchDataset):
    def __init__(self, df, sequence_length, src_cols, tgt_cols):
        """
        :param df: dataframe that holds data
        :param sequence_length: time series length for training
        :param src_cols: list of column names for source
        :param tgt_cols: list of column names for target
        """
        # no need for a vectorizer
        super().__init__(df, vectorizer=None)
        self.sequence_length = sequence_length
        self.src_cols = src_cols
        self.tgt_cols = tgt_cols

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        row_idx = self._target_df.index[index]
        idx = list(self.df.index).index(row_idx)

        df1 = self.df[idx:idx+self.sequence_length]
        y_input = df1.loc[:, self.src_cols].to_numpy().astype(np.float32)

        return {"y_input": y_input}

