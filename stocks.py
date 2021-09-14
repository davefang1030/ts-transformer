from data import PdTorchDataset
import numpy as np


class StockDataset(PdTorchDataset):
    def __init__(self, df, lookback_window, forward_window):
        super().__init__(df, vectorizer=None)
        self.lookback_window = lookback_window
        self.forward_window = forward_window

    def __getitem__(self, index):
        row_idx = self._target_df.index[index]
        idx = list(self.df.index).index(row_idx)
        df1 = self.df[idx:idx+self.lookback_window-1]
        x_source = np.array(df1.loc[:, df1.columns != 'split'])
        df1 = self.df[idx+self.lookback_window:idx+self.lookback_window+self.forward_window-1]
        y_target = np.array(df1.loc[:, df1.columns != 'split'])

        return {"x_source": x_source,
                "x_target": y_target}

