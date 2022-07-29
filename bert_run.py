import pandas as pd
import numpy as np
from itertools import product
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import time
import random
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from bert_data import DistilBertData
from bert_model import DistilBertClf, train, evaluate
from utils import custom_train_test_split, get_lists


def set_seed(random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


if __name__ == "__main__":
    # UNCOMMENT CHOSEN DATASET
    DATASET_NAME = 'kaggle'
    # DATASET_NAME = 'cmu'

    data = pd.read_pickle('./data/' + DATASET_NAME + '_BERT.pkl')

    # binarize labels
    mlb = MultiLabelBinarizer()
    y_binarized = mlb.fit_transform(data['genres'])
    label_names = [c for c in mlb.classes_]
    # save label names for inference, already done
    #joblib.dump(label_names, './labels_' + DATASET_NAME + '.pkl')

    X_train_total, y_train_total, X_test, y_test = custom_train_test_split(data['plots'], y_binarized, test_size=0.2, order=1)
    print('First data split done')

    # 70% training, 10% validation, 20% testing
    X_train, y_train, X_val, y_val = custom_train_test_split(X_train_total, y_train_total, test_size=0.125, order=1)
    print('Second data split done')

    set_seed()
    EPOCHS = 10
    MAX_LEN = 250   # best CMU model with MAX_LEN=512
    NUM_LABELS = y_train.shape[1]

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Dataset stores samples and corresponding labels
    train_data = DistilBertData(X=X_train, y=y_train, tokenizer=tokenizer, max_token_len=MAX_LEN)
    val_data = DistilBertData(X=X_val, y=y_val, tokenizer=tokenizer, max_token_len=MAX_LEN)
    test_data = DistilBertData(X=X_test, y=y_test, tokenizer=tokenizer, max_token_len=MAX_LEN)

    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f'\nDevice is {device}.\n')

    '''
    only for hyperparameter estimation use multiple 'params' values and set testing_mode=False:
    structure: batch size, learning rate
    
    learning rates as suggested in BERT paper
    params = [[16], [2e-5, 3e-5, 5e-5]]
    testing_mode = False
    '''
    # best Kaggle model ran with lr = 2e-5, best CMU model with lr = 5e-5
    params = [[16], [2e-5]]
    hyperparams = product(*params)
    testing_mode = True

    for batch_size, lr in hyperparams:
        # DataLoader wraps iterable around Dataset
        train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_data, batch_size=batch_size)
        test_dl = DataLoader(test_data, batch_size=batch_size)

        model = DistilBertClf(bert_model=bert_model, num_labels=NUM_LABELS)
        model.to(device)

        # for TensorBoard
        comment = f' DISTILBERT {DATASET_NAME} batch_size={batch_size} lr={lr}'
        tb = SummaryWriter(comment=comment)

        # prepare emtpy lists for metrics
        train_metrics_all_epochs = get_lists(7)
        val_metrics_all_epochs = get_lists(7)

        # initialize loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=lr)
        total_train_steps = len(train_dl) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_train_steps)

        # initialize model state and min validation loss
        best_model_state = copy.deepcopy(model.state_dict())
        best_val_loss = float('inf')
        best_epoch = 0

        # start epochs
        start = time.time()
        for epoch in range(EPOCHS):
            print('-' * 70)
            print(f'BERT EPOCH {epoch + 1}\n' + '-' * 70)

            train_metrics = train(train_dl, model, criterion, optimizer, scheduler)
            val_metrics, _ = evaluate(val_dl, model, criterion)

            # gather epoch metrics
            for met_list, met in zip(train_metrics_all_epochs, train_metrics):
                met_list.append((met, epoch))
            for met_list, met in zip(val_metrics_all_epochs, val_metrics):
                met_list.append((met, epoch))

            # save best model state w.r.t validation loss
            current_val_loss = val_metrics[0][1]
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch

            print(
                f'TRAIN LOSS: {train_metrics[0][1]:.4f}, TRAIN JACCS: {train_metrics[2][1]:.2f}, TRAIN F1S: {train_metrics[3][1]:.2f}')
            print(
                f'VAL LOSS: {val_metrics[0][1]:.4f}, VAL JACCS: {val_metrics[2][1]:.2f}, VAL F1S: {val_metrics[3][1]:.2f}')

        # format time
        end = time.time()
        elapsed = end - start
        hours = f'{elapsed // 3600:.0f}'
        minutes = f'{(elapsed - (3600 * int(hours))) // 60:.0f}'
        remain = f'{(elapsed - (3600 * int(hours))):.0f}'
        seconds = f'{int(remain) - (60 * int(minutes)):.0f}'
        print(f'Finished running {EPOCHS} epochs in', hours + 'h', minutes + 'min', seconds + 'sec.')
        print(f'Best epoch: {best_epoch}')

        # load best model weights and save model for inference, already done
        model.load_state_dict(best_model_state)
        #torch.save(model.state_dict(),
        #           './models/distilbert_checkpoint_' + DATASET_NAME + '_' + str(batch_size) + '_' + str(lr) + '.pth')

        # save scalars for TensorBoard
        for met_list in train_metrics_all_epochs:
            for (name, metric), epoch in met_list:
                tb.add_scalar(name + '/Training', metric, epoch)
        for met_list in val_metrics_all_epochs:
            for (name, metric), epoch in met_list:
                tb.add_scalar(name + '/Validation', metric, epoch)

        # save metrics
        joblib.dump(train_metrics_all_epochs,
                    './distilbert_train_metrics_' + DATASET_NAME + '_' + str(batch_size) + '_' + str(lr) + '.pkl')
        joblib.dump(val_metrics_all_epochs,
                    './distilbert_val_metrics_' + DATASET_NAME + '_' + str(batch_size) + '_' + str(lr) + '.pkl')

        if testing_mode:
            # final test on test set
            test_metrics, reports = evaluate(test_dl, model, criterion, testing=True, label_names=label_names)
            # report to see F1 scores per label
            reports_df = pd.concat(reports)
            reports_mean = reports_df.groupby(reports_df.index).mean()

            joblib.dump(test_metrics,
                        './distilbert_test_metrics_' + DATASET_NAME + '_' + str(batch_size) + '_' + str(lr) + '.pkl')
            joblib.dump(reports_mean,
                        './distilbert_test_report_' + DATASET_NAME + '_' + str(batch_size) + '_' + str(lr) + '.pkl')

            print('-' * 70)
            print(f'TEST LOSS: {test_metrics[0][1]:.4f}, TEST JACCS: {test_metrics[2][1]:.2f}, TEST F1S: {test_metrics[3][1]:.2f}')
    tb.close()
