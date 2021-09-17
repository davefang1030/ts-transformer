from data import PdTorchDataset
import numpy as np


class StockDataset(PdTorchDataset):
    def __init__(self, df, lookback_window, forward_window):
        """
        No need of a vectorizer
        :param df:
        :param lookback_window:
        :param forward_window:
        """
        super().__init__(df, vectorizer=None)
        self.lookback_window = lookback_window
        self.forward_window = forward_window

    def __getitem__(self, index):
        """
        starting from index, there are lookback_window obs as input, followed by forward_window obs as target
        :param index:
        :return:
        """
        row_idx = self._target_df.index[index]
        idx = list(self.df.index).index(row_idx)
        df1 = self.df[idx:idx+self.lookback_window]
        x_source = df1.loc[:, df1.columns != 'split'].to_numpy()
        # remove date column, also change dtype to float32. because of date column, dtype is object
        x_source = x_source[:, 1:].astype(np.float32)
        df1 = self.df[idx+self.lookback_window+1:idx+self.lookback_window+self.forward_window+1]
        y_target = df1.loc[:, df1.columns != 'split'].to_numpy()
        # remove date column
        y_target = y_target[:, 1:].astype(np.float32)

        return {"x_source": x_source,
                "y_target": y_target}

