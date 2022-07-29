from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import unicodedata
import numpy as np
import pandas as pd
import re
import ast
# from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification


#### GENERAL DATA PRE-PROCESSING ####
# transform genre dictionaries to list of genres
def genres_to_list_kaggle(df):
    genre_list = []
    for g in df['genres']:
        literal = ast.literal_eval(g)
        # for each row make new list of target genres
        genres = []
        for dictionary in literal:
            genres.append(dictionary['name'])
        genre_list.append(list(genres))
    return genre_list


def genres_to_list_cmu(df):
    genre_list = []
    for g in df['genres']:
        literal = ast.literal_eval(g)
        genre_list.append(list(literal.values()))
    return genre_list


# remove rows without genres
def remove_empty_genres_kaggle(df):
    df_new = df[df['genres'] != "[]"]
    df_new = df_new.reset_index(drop=True)
    return df_new


def remove_empty_genres_cmu(df):
    df_new = df[df['genres'].str.len() != 0]
    df_new = df_new.reset_index(drop=True)
    return df_new


# number of distinct genres
def number_genres(genre_list):
    all_genres = []
    for genres in genre_list:
        for genre in genres:
            all_genres.append(genre)
    print(f'Number of distinct genres: {len(set(all_genres))}')
    return len(set(all_genres))


# find rows with production company genres in Kaggle
def remove_non_genres_kaggle(df):
    genre_list = genres_to_list_kaggle(df)
    idx_to_remove = []
    false_genres = ['The Cartel', 'Carousel Productions', 'GoHands', 'Aniplex', 'Rogue State', 'Pulser Productions',
                    'Mardock Scramble Production Committee', 'Telescene Film Group Productions', 'Odyssey Media',
                    'Sentai Filmworks',
                    'Vision View Entertainment', 'BROSTA TV']
    for i, j in enumerate(genre_list):
        if any(fg in j for fg in false_genres):
            # print(i, j)
            idx_to_remove.append(i)
    df_new = df.drop(idx_to_remove)
    df_new = df_new.reset_index(drop=True)
    return df_new


def remove_duplicates(df):
    df_new = df.drop_duplicates(subset='plots', keep='first')
    df_new = df_new.reset_index(drop=True)
    return df_new


#### DATA CLEANING ####
def tokenization(text):
    wt = ' '.join(word_tokenize(text))
    return wt


# remove_accents function adapted from:
# https://github.com/shashankvmaiya/Movie-Genre-Multi-Label-Text-Classification/blob/master/Movie_Genre_Classification.ipynb
def remove_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return text


def do_spaces(text):
    # remove multiple whitespaces
    text = re.sub(r'\s+', ' ', text)
    # insert whitespace between letters and digits
    text = re.sub(r'(\d+)([a-z])', r'\1 \2', text)
    text = re.sub(r'([a-z])(\d+)', r'\1 \2', text)
    return text


def remove_stopwords(text):
    stop = stopwords.words('english')
    stop.append('ca')
    stop.append('nt')
    stop.append('n')
    words = [word for word in text.split() if word not in stop]
    new_text = " ".join(words)
    return new_text


def remove_html(text):
    # custom html and url removal
    string_to_remove = r"ref name.[^}]\S+|http[^}]\S+|cite web[^}][^>]*|cite book|cquote|expand section[\|\S]*|empty section|\{quotation\}|cite web|\|\S+"
    text = text.lower()
    # remove html etc.
    text = re.sub(string_to_remove, "", text)
    text = ' '.join(text.split())
    return text


def clean_text(text):
    # convert text to lowercase
    text = text.lower()
    # keep only alphanumeric
    text = re.sub(r"[^a-zA-Z0-9\s]", '', text)
    # remove whitespaces
    text = ' '.join(text.split())
    return text


def clean_text_bert(text):
    # convert text to lowercase
    text = text.lower()
    # keep only alphanumeric and basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s\-\.,!?\"'\$]", '', text)
    # remove whitespaces
    text = ' '.join(text.split())
    return text


def clean_kaggle(df, col):
    new_col = col + '_clean'
    df[new_col] = df[col].apply(lambda x: remove_accents(x))
    df[new_col] = df[new_col].apply(lambda x: tokenization(x))
    df[new_col] = df[new_col].apply(lambda x: clean_text(x))
    df[new_col] = df[new_col].apply(lambda x: do_spaces(x))
    df[new_col] = df[new_col].apply(lambda x: remove_stopwords(x))
    return df


# for BERT do not remove stopwords or punctuation, no tokenizing
def clean_kaggle_bert(df, col):
    new_col = col + '_clean'
    df[new_col] = df[col].apply(lambda x: remove_accents(x))
    df[new_col] = df[new_col].apply(lambda x: clean_text_bert(x))
    df[new_col] = df[new_col].apply(lambda x: do_spaces(x))
    return df


def clean_cmu(df, col):
    new_col = col + '_clean'
    df[new_col] = df[col].apply(lambda x: remove_html(x))
    df[new_col] = df[new_col].apply(lambda x: remove_accents(x))
    df[new_col] = df[new_col].apply(lambda x: tokenization(x))
    df[new_col] = df[new_col].apply(lambda x: clean_text(x))
    df[new_col] = df[new_col].apply(lambda x: do_spaces(x))
    df[new_col] = df[new_col].apply(lambda x: remove_stopwords(x))
    return df


# for BERT do not remove stopwords or punctuation, no tokenizing
def clean_cmu_bert(df, col):
    new_col = col + '_clean'
    df[new_col] = df[col].apply(lambda x: remove_html(x))
    df[new_col] = df[new_col].apply(lambda x: remove_accents(x))
    df[new_col] = df[new_col].apply(lambda x: clean_text_bert(x))
    df[new_col] = df[new_col].apply(lambda x: do_spaces(x))
    return df


def clean_for_inference(text):
    text = remove_accents(text)
    text = tokenization(text)
    text = clean_text(text)
    text = do_spaces(text)
    text = remove_stopwords(text)
    return text


def clean_for_inference_bert(text):
    text = remove_accents(text)
    text = clean_text_bert(text)
    text = do_spaces(text)
    return text


#### STATISTICS ####
def avg_max_min_plot_len(df, col):
    num_words = []
    for plot in df[col]:
        num_words.append(len(plot.split()))
    avg_words = sum(num_words) / len(num_words)
    maxi = max(num_words)
    mini = min(num_words)
    print(f'Average plot length: {round(avg_words, 1)} \nLongest plot: {maxi} \nShortest plot: {mini}')


def unique_tokens(df, col):
    all_tokens = []
    for plot in df[col]:
        for word in plot.split():
            all_tokens.append(word)
    print(f'Number of unique tokens: {len(set(all_tokens))}')


def lcard_lden(df, num_genres):
    len_genres = []
    for genre_set in df['genres']:
        len_genres.append(len(genre_set))
    avg_genres = sum(len_genres) / len(len_genres)
    label_density = avg_genres / num_genres
    print(f'Label Cardinality (avg. number of genres): {round(avg_genres, 1)}')
    print(f'Label Density: {round(label_density, 4)}')


def distinct_label_sets(df):
    unique = []
    for genre_set in df['genres']:
        frozen_genres = frozenset(genre_set)
        if frozen_genres not in unique:
            unique.append(frozen_genres)
    unique_str = []
    for u in unique:
        u = str(u)
        unique_str.append(u)
    dls = len(unique_str)
    avg_dls = len(df) / dls
    prop_dls = dls / len(df)
    print(f'DistL: {dls},\n'
          f'Avg. frequency of DistL: {round(avg_dls, 1)},\n'
          f'Proportion of DistL: {round(prop_dls, 4)}')


def print_kaggle_stats(df, num_genres):
    print('Kaggle statistics:')
    print('')
    avg_max_min_plot_len(df, 'plots')
    unique_tokens(df, 'plots')
    lcard_lden(df, num_genres)
    distinct_label_sets(df)
    print('')


def print_cmu_stats(df, num_genres):
    print('CMU statistics:')
    print('')
    avg_max_min_plot_len(df, 'plots')
    unique_tokens(df, 'plots')
    lcard_lden(df, num_genres)
    distinct_label_sets(df)
    print('')


#### SPLITTING ####
def stratify(X, y, test_size, order):
    # iterative_train_test_split from skmultilearn.model_selection,
    # customized to include order
    stratifier = IterativeStratification(n_splits=2, order=order,
                                         sample_distribution_per_fold=[test_size, 1.0 - test_size])
    train_indexes, test_indexes = next(stratifier.split(X, y))
    X_train, y_train = X[train_indexes, :], y[train_indexes, :]
    X_test, y_test = X[test_indexes, :], y[test_indexes, :]
    return X_train, y_train, X_test, y_test


def custom_train_test_split(X, y, test_size, order):
    # input needs to be matrix
    X = np.asmatrix(pd.DataFrame(X))
    y = np.asmatrix(y)
    # split data with stratification
    X_train, y_train, X_test, y_test = stratify(X, y, test_size=test_size, order=order)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # output as array
    X_train = np.array([plot for sublist in np.asarray(X_train) for plot in sublist])
    y_train = np.asarray(y_train)
    X_test = np.array([plot for sublist in np.asarray(X_test) for plot in sublist])
    y_test = np.asarray(y_test)
    return X_train, y_train, X_test, y_test


def get_lists(n):
    # return a list of n empty lists
    empty_lists = [[] for _ in range(n)]
    return empty_lists
