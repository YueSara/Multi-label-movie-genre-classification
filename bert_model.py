import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from utils import get_lists


class DistilBertClf(nn.Module):
    def __init__(self, bert_model, num_labels, freeze_bert=False):
        super(DistilBertClf, self).__init__()
        self.num_labels = num_labels
        self.bert = bert_model
        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, self.num_labels)
        )

        # if just one dense layer after BERT instead of two:
        # self.classifier = nn.Linear(hidden_size, num_labels)

        # if freeze_bert=False, BERT is fine-tuned
        # if freeze_bert=True, use BERT as fixed feature extractor, only last classifier is trained
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)

        # use only CLS tokens of the outputs
        # pooled shape = (batch, bert.config.hiddensize) --> (16, 768)
        pooled = outputs[0][:, 0, :]

        # logits shape = (batch, num_labels) --> (16, 20/363)
        logits = self.classifier(pooled)

        return logits


def get_metrics(y_true, pred, thresh):
    pred = torch.sigmoid(pred)
    y_true = y_true.detach().to('cpu').numpy()
    pred = pred.detach().to('cpu').numpy()

    acc = round(metrics.accuracy_score(y_true, pred > thresh), 6),
    jaccs = round(metrics.jaccard_score(y_true, pred > thresh, average='samples'), 6)

    report = pd.DataFrame(metrics.classification_report(y_true, pred > thresh, output_dict=True,
                                                        zero_division=0)).T
    f1s = round(report.loc['samples avg', 'f1-score'], 6)
    ps = round(report.loc['samples avg', 'precision'], 6)
    rs = round(report.loc['samples avg', 'recall'], 6)
    ham = round(metrics.hamming_loss(y_true, pred > thresh), 6)

    metrics_per_batch = [acc, jaccs, f1s, ps, rs, ham]

    return metrics_per_batch


# trains one epoch
def train(train_dl, model, criterion, optimizer, scheduler):
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    train_metrics_all_batches = get_lists(7)
    all_metrics = []

    model.train()  # sets training mode

    for step, batch in enumerate(train_dl):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        pred = model(input_ids, attention_mask)

        loss = criterion(pred, labels)
        metrics_per_batch = get_metrics(labels, pred, thresh=0.5)
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
        scheduler.step()

        if step % 500 == 0:
            print(f'batch: {step}, loss: {loss.item():>7f},'
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
        for _, batch in enumerate(test_dl):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            pred = model(input_ids, attention_mask)

            loss = criterion(pred, labels)

            metrics_per_batch = get_metrics(labels, pred, thresh=0.5)
            metrics_per_batch.insert(0, loss.item())

            # put each metric in its own list
            for met_list, met in zip(val_metrics_all_batches, metrics_per_batch):
                met_list.append(met)

            if testing:
                pred = torch.sigmoid(pred)
                labels = labels.detach().to('cpu').numpy()
                pred = pred.detach().to('cpu').numpy()

                report = pd.DataFrame(metrics.classification_report(labels, pred >= 0.5, output_dict=True,
                                                                    zero_division=0, target_names=label_names)).T
                reports.append(report)

    # for each metric get mean over all batches, append metric name for access in TensorBoard
    m_names = ['Loss', 'Subset_Accuracy', 'Jaccard Samples', 'F1 Samples', 'Precision Samples', 'Recall Samples',
               'Hamming Loss']
    for met_list, name in zip(val_metrics_all_batches, m_names):
        mean = np.mean(met_list)
        all_metrics.append((name, mean))

    return all_metrics, reports
