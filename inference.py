import torch
import re
import joblib
import numpy as np
from utils import clean_for_inference, clean_for_inference_bert
from bilstm_model import BilstmModel
from bilstm_run import get_embedding_matrix
from bert_model import DistilBertClf
from transformers import DistilBertModel, DistilBertTokenizer


#### BASELINE ####
def infer_from_baseline(input_text):
    # baseclf for Kaggle is Classifier Chain, for CMU is RAkEL D
    tfidf = joblib.load('./models/tfidf_' + DATASET_NAME + '.pkl')
    clf = joblib.load('./models/baseclf_' + DATASET_NAME + '.pkl')

    if DATASET_NAME == 'kaggle':
        c = 'Classifier Chain'
    elif DATASET_NAME == 'cmu':
        c = 'RAkEL'

    print(f'Classifier: {c}')

    # clean and tf-idf each plot, predict genres
    for i, t in enumerate(input_text):
        t = clean_for_inference(t)
        text_tfidf = tfidf.transform([t])
        prediction = clf.predict(text_tfidf)
        prediction = np.array(prediction.todense())[0]

        print(f'Predicted classes for plot {i+1}: ')
        for label, pred in zip(label_names, prediction):
            if pred:
                print(f'\t{label}')


#### BiLSTM ####
def infer_from_bilstm(input_text, thresh):
    # num layers is 1 for Kaggle's model, 2 for CMU's model
    if DATASET_NAME == 'kaggle':
        layers = 1
        state_dict_path = './models/bilstm_checkpoint_kaggle_adamw_8_256_0.001_1.pth'
    elif DATASET_NAME == 'cmu':
        layers = 2
        state_dict_path = './models/bilstm_checkpoint_cmu_adamw_8_256_0.001_2_len145.pth'

    # load tokenizer, get embedding matrix
    tokenizer = joblib.load('./models/bilstm_tokenizer_'+DATASET_NAME+'_len145.pkl')
    word_index = tokenizer.word_index
    vocab_size = len(word_index)
    embedding_matrix = get_embedding_matrix(word_index, embedding_dim=300)

    # load model
    model = BilstmModel(input_size=300, hidden_size=256, output_size=num_labels, vocab_size=vocab_size,
                        embedding_matrix=embedding_matrix, layers=layers)
    model.load_state_dict(torch.load(state_dict_path))

    # clean and tokenize each plot, predict genres
    print(f'Classifier: BiLSTM')
    for i, t in enumerate(input_text):
        # process input text
        t = clean_for_inference(t)
        sequence = tokenizer.texts_to_sequences([t])
        sequence = torch.tensor(sequence)

        model.eval()
        with torch.no_grad():
            logits = model(sequence)
        probs = torch.sigmoid(logits).to('cpu').numpy()

        predicted = probs[0] >= thresh

        print(f'Predicted classes for plot {i+1}: ')
        for label, pred in zip(label_names, predicted):
            if pred:
                print(f'\t{label}')


#### BERT ####
def infer_from_bert(input_text, thresh):
    if DATASET_NAME == 'kaggle':
        state_dict_path = './models/distilbert_checkpoint_kaggle_16_2e-05.pth'
    elif DATASET_NAME == 'cmu':
        state_dict_path = './models/distilbert_checkpoint_cmu_16_5e-05_512.pth'

    # initialize and load model and tokenizer
    distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model = DistilBertClf(bert_model=distilbert, num_labels=num_labels)
    model.load_state_dict(torch.load(state_dict_path))
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # clean and tokenize each plot, predict genres
    print(f'Classifier: BERT')
    for i, t in enumerate(input_text):
        t = clean_for_inference_bert(t)
        inputs = tokenizer(t, padding=True, return_tensors='pt')

        model.eval()
        with torch.no_grad():
            logits = model(inputs['input_ids'], inputs['attention_mask'])
        probs = torch.sigmoid(logits).to('cpu').numpy()

        predicted = probs[0] >= thresh

        print(f'Predicted classes for plot {i+1}: ')
        for label, pred in zip(label_names, predicted):
            if pred:
                print(f'\t{label}')


#### INFERENCE WITH TRAINED MODELS ####
if __name__ == "__main__":
    # UNCOMMENT CHOSEN DATASET
    DATASET_NAME = 'kaggle'
    #DATASET_NAME = 'cmu'

    # load labels and input path
    label_names = joblib.load('labels_' + DATASET_NAME + '.pkl')
    num_labels = len(label_names)
    input_path = 'inference_input_text.txt'

    # prepare input plot(s)
    input_texts = []
    f = open(input_path, 'r')
    texts = f.read()
    texts = texts.split("\n\n")
    for text in texts:
        input_texts.append(re.sub('\n', '', str(text)))

    # UNCOMMENT CHOSEN METHOD
    infer_from_baseline(input_texts)
    #infer_from_bilstm(input_texts, thresh=0.5)
    #infer_from_bert(input_texts, thresh=0.5)