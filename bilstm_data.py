import torch
from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences

class BilstmData(Dataset):
    def __init__(self, X, y):
        super(BilstmData, self).__init__()
        # X is padded sequences
        # y is y_train/val/test
        self.plots = X
        self.genres = y

    def __len__(self):
        return len(self.plots)

    def __getitem__(self, idx):
        plot = torch.Tensor(self.plots[idx]).to(torch.int64)
        genres = torch.Tensor(self.genres[idx])

        return plot, genres


def process_data_for_bilstm(X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, max_len):
    # text to sequences of indices
    sequences_train = tokenizer.texts_to_sequences(X_train)
    sequences_val = tokenizer.texts_to_sequences(X_val)
    sequences_test = tokenizer.texts_to_sequences(X_test)

    # pad sequences
    X_train_padded = torch.Tensor(pad_sequences(sequences_train, maxlen=max_len))
    X_val_padded = torch.Tensor(pad_sequences(sequences_val, maxlen=max_len))
    X_test_padded = torch.Tensor(pad_sequences(sequences_test, maxlen=max_len))

    # Dataset stores samples and corresponding labels
    train_data = BilstmData(X=X_train_padded, y=y_train)
    val_data = BilstmData(X=X_val_padded, y=y_val)
    test_data = BilstmData(X=X_test_padded, y=y_test)

    return train_data, val_data, test_data