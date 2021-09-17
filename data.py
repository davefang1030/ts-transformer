from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json


class PdTorchDataset(Dataset):
    """ base class for pd dataframe and torch dataset utility """

    def __init__(self, df, vectorizer):
        """

        :param df: panda dataframe
        :param vectorizer: vectorizer to convert vocabulary to indices
        """
        super().__init__()
        self.df = df
        self._vectorizer = vectorizer

        # set three datasets
        self.train_df = self.df[self.df.split == 'train']
        self.train_size = len(self.train_df)
        self.val_df = self.df[self.df.split == 'val']
        self.validation_size = len(self.val_df)
        self.test_df = self.df[self.df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}
        # set initial dataset to train
        self.set_split('train')

        # class weights to counter inbalanced dataset. Basically user 1/class_size as weights for CrossEntropyLoss
        self._calculate_weights()

    def _calculate_weights(self):
        """
        By default we set class_weights to None as we don't really care about weights
        :return:
        """
        self.class_weights = None
        self.num_class = 0

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def save_vectorizer(self, vectorizer_filepath):
        """saves the vectorizer to disk using json
        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size

    def __getitem__(self, index):
        raise NotImplementedError

    @staticmethod
    def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
        """
        A generator function that wraps PyTorchData
        :param dataset: PdTorchDataset,
        :param batch_size:
        :param shuffle:
        :param drop_last:
        :param device:
        :return:
        """
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                                shuffle=shuffle, drop_last=drop_last)

        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict

    @classmethod
    def load_dataset_and_make_vectorizer(cls, csv, vectorizer_cls):
        """

        :param csv: str, file path of the dataset
        :param vectorizer_cls: vectorizer class name
        :return: instance of dataset
        """
        csv_df = pd.read_csv(csv)
        vectorizer = vectorizer_cls.from_dataframe(csv_df)
        return cls(csv_df, vectorizer)

