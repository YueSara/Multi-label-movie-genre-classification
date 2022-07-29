import torch
from torch.utils.data import Dataset

class DistilBertData(Dataset):
    def __init__(self, X, y, tokenizer, max_token_len):
        super(DistilBertData, self).__init__()
        # X is X_train/val/test: array of strings
        # y is y_train/val/test: binary array of len=NUM_LABELS
        self.plots = X
        self.genres = y
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.plots)

    def __getitem__(self, idx):
        plot = self.plots[idx]
        labels = self.genres[idx]

        # tokenizer() returns input ids and attention mask as dict
        # attention mask is 1 if it's a word and 0 if it's padding
        # position ids created automatically, token type (segment) ids not needed bc only passing one seq at a time
        # also distilbert does not take token type ids anyway
        inputs = self.tokenizer(plot,
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_token_len,
                                return_tensors='pt'
                                )

        encoding = dict(
            input_ids=inputs['input_ids'].flatten(),
            attention_mask=inputs['attention_mask'].flatten(),
            labels=torch.Tensor(labels).to(torch.float)
        )

        return encoding



