import pandas as pd
import numpy as np
import time
import random
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
from itertools import product
import joblib
import copy
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from bilstm_model import BilstmModel, train, evaluate
from bilstm_data import process_data_for_bilstm
from utils import custom_train_test_split, get_lists


def set_seed(random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def get_embedding_matrix(word_index, embedding_dim):
    unique_words = len(word_index) + 1
    skipped_words = 0
    embedding_matrix = np.zeros((unique_words, embedding_dim))

    word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=100000)

    for word, index in word_index.items():
        embedding_vector = None
        try:
            embedding_vector = word2vec[word]
        except:
            skipped_words += 1
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix


if __name__ == "__main__":
    # UNCOMMENT CHOSEN DATASET
    DATASET_NAME = 'kaggle'
    #DATASET_NAME = 'cmu'
    max_len = 145

    data = pd.read_pickle('./data/' + DATASET_NAME + '.pkl')

    if DATASET_NAME == 'cmu':
        # truncate CMU plots to max_len
        truncated = data.plots.apply(lambda x: x.split()[:max_len])
        truncated_df = pd.DataFrame({'plots': truncated})
        truncated_df = truncated_df.plots.apply(lambda x: ' '.join(x))
        data['plots'] = truncated_df

    # binarize labels
    mlb = MultiLabelBinarizer()
    y_binarized = mlb.fit_transform(data['genres'])
    label_names = [c for c in mlb.classes_]
    # save label names for inference, already done
    #joblib.dump(label_names, './labels_' + DATASET_NAME + '.pkl')

    X_train_total, y_train_total, X_test, y_test = custom_train_test_split(data['plots'], y_binarized, test_size=0.2, order=1)
    print('First data split done')

    # use train + val set to build vocab
    tokenizer = Tokenizer(num_words=100000)
    tokenizer.fit_on_texts(X_train_total)
    # save label names for inference, already done
    #joblib.dump(tokenizer, './models/bilstm_tokenizer_' + DATASET_NAME + '_len' + str(max_len) + '.pkl')

    # 70% training, 10% validation, 20% testing
    # 0.8 * 0.125 = 0.1
    X_train, y_train, X_val, y_val = custom_train_test_split(X_train_total, y_train_total, test_size=0.125, order=1)
    print('Second data split done')

    set_seed()
    EPOCHS = 10
    EMBEDDING_DIM = 300
    NUM_LABELS = y_train.shape[1]
    VOCAB_SIZE = len(tokenizer.word_index)

    embedding_matrix = get_embedding_matrix(tokenizer.word_index, EMBEDDING_DIM)
    train_data, val_data, test_data = process_data_for_bilstm(X_train, y_train, X_val, y_val, X_test, y_test,
                                                              tokenizer, max_len)

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f'Device is {device}.')

    '''
    # only for hyperparameter estimation use multiple 'params' values and set testing_mode=False:
    # structure: optimizer name, batch size, hidden size, learning rate, number of bilstm layers
    
    params = [['adam', 'adamw', 'sgd'], [8, 16, 32], [64, 128, 256], [1e-2, 1e-3, 1e-4], [1, 2]]
    testing_mode = False
    '''

    # best Kaggle model ran with num_layers = 1, best CMU model with num_layers = 2
    params = [['adamw'], [8], [256], [1e-3], [1]]
    hyperparams = product(*params)
    testing_mode = True

    for optim_name, batch_size, hidden_size, lr, num_layers in hyperparams:
        # DataLoader wraps iterable around Dataset
        train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_data, batch_size=batch_size)
        test_dl = DataLoader(test_data, batch_size=batch_size)

        model = BilstmModel(
            input_size=EMBEDDING_DIM,  # 300
            hidden_size=hidden_size,
            output_size=NUM_LABELS,  # 20 or 363
            vocab_size=VOCAB_SIZE,
            embedding_matrix=embedding_matrix,
            layers=num_layers
        )
        model.to(device)

        # for TensorBoard
        comment = f' BILSTM {DATASET_NAME} optimizer={optim_name} batch_size={batch_size} hidden_size={hidden_size} lr={lr} layers={num_layers} len={max_len}'
        tb = SummaryWriter(comment=comment)

        # prepare emtpy lists for metrics
        train_metrics_all_epochs = get_lists(7)
        val_metrics_all_epochs = get_lists(7)

        criterion = nn.BCEWithLogitsLoss()

        # get chosen optimizer
        if optim_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optim_name == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        elif optim_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

        # initialize model state and min validation loss
        best_model_state = copy.deepcopy(model.state_dict())
        best_val_loss = float('inf')
        best_epoch = 0

        # start epochs
        start = time.time()
        for epoch in range(EPOCHS):
            print('-' * 70)
            print(f'BILSTM EPOCH {epoch + 1}\n' + '-' * 70)

            train_metrics = train(train_dl, model, criterion, optimizer)
            val_metrics, _ = evaluate(val_dl, model, criterion)

            if optim_name == 'sgd':
                # val_metrics[0][1] is validation loss of current epoch
                scheduler.step(val_metrics[0][1])

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
        #           './models/bilstm_checkpoint_' + DATASET_NAME + '_' + optim_name + '_' + str(
        #               batch_size) + '_' + str(hidden_size) + '_' + str(lr) + '_' + str(num_layers) + '_len' + str(max_len) + '.pth')

        # save scalars for TensorBoard
        for met_list in train_metrics_all_epochs:
            for (name, metric), epoch in met_list:
                tb.add_scalar(name + '/Training', metric, epoch)
        for met_list in val_metrics_all_epochs:
            for (name, metric), epoch in met_list:
                tb.add_scalar(name + '/Validation', metric, epoch)

        # save metrics
        joblib.dump(train_metrics_all_epochs,
                    './bilstm_train_metrics_' + DATASET_NAME + '_' + optim_name + '_' + str(batch_size) + '_' + str(
                        hidden_size) + '_' + str(lr) + '_' + str(num_layers) + '_len' + str(max_len) + '.pkl')
        joblib.dump(val_metrics_all_epochs,
                    './bilstm_val_metrics_' + DATASET_NAME + '_' + optim_name + '_' + str(batch_size) + '_' + str(
                        hidden_size) + '_' + str(lr) + '_' + str(num_layers) + '_len' + str(max_len) + '.pkl')

        if testing_mode:
            # final test on test set
            test_metrics, reports = evaluate(test_dl, model, criterion, testing=True, label_names=label_names)
            # report to see F1 scores per label
            reports_df = pd.concat(reports)
            reports_mean = reports_df.groupby(reports_df.index).mean()

            joblib.dump(test_metrics,
                        './bilstm_test_metrics_' + DATASET_NAME + '_' + optim_name + '_' + str(batch_size) + '_' + str(
                            hidden_size) + '_' + str(lr) + '_' + str(num_layers) + '_len' + str(max_len) + '.pkl')
            joblib.dump(reports_mean,
                        './bilstm_test_report_' + DATASET_NAME + '_' + optim_name + '_' + str(batch_size) + '_' + str(
                            hidden_size) + '_' + str(lr) + '_' + str(num_layers) + '_len' + str(max_len) + '.pkl')

            print('-' * 70)
            print(
                f'TEST LOSS: {test_metrics[0][1]:.4f}, TEST JACCS: {test_metrics[2][1]:.2f}, TEST F1S: {test_metrics[3][1]:.2f}')
    tb.close()
