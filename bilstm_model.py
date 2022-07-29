import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sklearn.metrics as metrics
from utils import get_lists


class BilstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, embedding_matrix, layers):
        super(BilstmModel, self).__init__()
        self.input_size = input_size  # embedding size e.g. 300
        self.hidden_size = hidden_size  # e.g. [64, 256]
        self.output_size = output_size  # output size = number of labels (20 or 363)
        self.vocab_size = vocab_size
        self.embedding_matrix = torch.Tensor(embedding_matrix)

        self.num_directions = 2
        self.num_layers = layers

        # Embedding Layer
        self.embeddings = nn.Embedding(self.vocab_size, self.input_size)
        self.embeddings.weight = nn.Parameter(self.embedding_matrix, requires_grad=False)

        # BiLSTM Layer
        self.bilstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.5,
            bidirectional=True,
            batch_first=False
        )
        self.fc = nn.Linear(self.hidden_size * self.num_directions, self.output_size)

    def forward(self, x):
        # x shape = (batch_size, max_len)
        embedded = self.embeddings(x)
        # transpose embedded shape from (batch_size, max_len, input_size) to (max_len, batch_size, input_size)
        embedded = torch.transpose(embedded, 0, 1)

        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'

        # initialize hidden and cell state
        h_0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(device)
        bilstm_out, (h_n, c_n) = self.bilstm(embedded, (h_0, c_0))
        # take last seq for every batch when batch_first=False
        out = bilstm_out[-1, :, :]
        final_out = self.fc(out)

        return final_out


def get_metrics(y_true, pred, thresh=0.5):
    pred = torch.sigmoid(pred)
    y_true = y_true.detach().to('cpu').numpy()
    pred = pred.detach().to('cpu').numpy()

    acc = round(metrics.accuracy_score(y_true, pred >= thresh), 6),
    jaccs = round(metrics.jaccard_score(y_true, pred >= thresh, average='samples'), 6)

    report = pd.DataFrame(metrics.classification_report(y_true, pred >= thresh, output_dict=True,
                                                        zero_division=0)).T
    f1s = round(report.loc['samples avg', 'f1-score'], 6)
    ps = round(report.loc['samples avg', 'precision'], 6)
    rs = round(report.loc['samples avg', 'recall'], 6)
    ham = round(metrics.hamming_loss(y_true, pred >= thresh), 6)

    metrics_per_batch = [acc, jaccs, f1s, ps, rs, ham]

    return metrics_per_batch


# trains one epoch
def train(train_dl, model, criterion, optimizer):
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    train_metrics_all_batches = get_lists(7)
    all_metrics = []

    model.train()  # sets training mode

    for batch, (X, y) in enumerate(train_dl):
        X, y = X.to(device), y.to(device)

        pred = model(X)  # returns logits, raw predicted values [-inf, inf]

        loss = criterion(pred, y)
        metrics_per_batch = get_metrics(y, pred, thresh=0.5)
        metrics_per_batch.insert(0, loss.item())

        # put each metric in its own list
        for met_list, met in zip(train_metrics_all_batches, metrics_per_batch):
            met_list.append(met)

        # Backpropagation
        optimizer.zero_grad()
        # calculate gradient
        loss.backward()
        # adjust params with gradient descent (multiply LR and subtract step size from old params)
        optimizer.step()

        if batch % 500 == 0:
            print(f'batch: {batch}, loss: {loss.item():>7f},'
                  f'jaccs: {metrics_per_batch[2]:.4f}, f1s: {metrics_per_batch[3]:.4f}')

    # for each metric get mean over all batches, append metric name for access in TensorBoard
    m_names = ['Loss', 'Subset_Accuracy', 'Jaccard Samples', 'F1 Samples', 'Precision Samples', 'Recall Samples',
               'Hamming Loss']
    for met_list, name in zip(train_metrics_all_batches, m_names):
        mean = np.mean(met_list)
        all_metrics.append((name, mean))

    return all_metrics


# evaluates one epoch
def evaluate(test_dl, model, criterion, testing=False, label_names=None):
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    val_metrics_all_batches = get_lists(7)
    all_metrics = []

    # used when testing
    reports = []

    model.eval()  # sets eval mode, e.g. removes dropout

    with torch.no_grad():  # only forward, no backprop, sets requires_grad = False
        for X, y in test_dl:  # for each batch
            X, y = X.to(device), y.to(device)
            pred = model(X)

            loss = criterion(pred, y)

            metrics_per_batch = get_metrics(y, pred, thresh=0.5)
            metrics_per_batch.insert(0, loss.item())

            # put each metric in its own list
            for met_list, met in zip(val_metrics_all_batches, metrics_per_batch):
                met_list.append(met)

            if testing:
                pred = torch.sigmoid(pred)
                y = y.detach().to('cpu').numpy()
                pred = pred.detach().to('cpu').numpy()

                report = pd.DataFrame(metrics.classification_report(y, pred >= 0.5, output_dict=True,
                                                                    zero_division=0, target_names=label_names)).T
                reports.append(report)

    # for each metric get mean over all batches, append metric name for access in TensorBoard
    m_names = ['Loss', 'Subset_Accuracy', 'Jaccard Samples', 'F1 Samples', 'Precision Samples', 'Recall Samples',
               'Hamming Loss']
    for met_list, name in zip(val_metrics_all_batches, m_names):
        mean = np.mean(met_list)
        all_metrics.append((name, mean))

    return all_metrics, reports
