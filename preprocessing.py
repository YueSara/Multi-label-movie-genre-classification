import pandas as pd
from utils import *


KAGGLE_ORIGINAL_PATH = './data/movies_metadata.csv'
CMU_ORIGINAL_META = './data/movie.metadata.tsv'
CMU_ORIGINAL_PLOT = './data/plot_summaries.txt'


def kaggle_preprocessing(path=KAGGLE_ORIGINAL_PATH):
    #### LOAD DATA ####
    kaggle_original = pd.read_csv(path)
    kaggle = kaggle_original[['genres', 'overview', 'tagline', 'title']]
    kaggle = kaggle.rename(columns={"overview": "plots"})

    #### GENRE PROCESSING ####
    kaggle = remove_empty_genres_kaggle(kaggle)
    kaggle = remove_non_genres_kaggle(kaggle)

    # transform genres to list, set them as new target column
    genre_list = genres_to_list_kaggle(kaggle)
    genres_lower = [[g.lower() for g in genres] for genres in genre_list]
    kaggle['genres'] = genres_lower

    #### PLOT PROCESSING ####
    kaggle = kaggle.dropna(subset=['plots'])
    kaggle = remove_duplicates(kaggle)

    # create copy of data because of different cleaning process for BERT
    kaggle_bert = kaggle.copy()
    # clean plots
    kaggle = clean_kaggle(kaggle, 'plots')
    kaggle_bert = clean_kaggle_bert(kaggle_bert, 'plots')

    # remove plots that have become empty after cleaning
    # uses kaggle's empty plots for kaggle_bert
    # otherwise, because kaggle_bert has punctuation left, it would miss some empty plots
    empty = kaggle[kaggle['plots_clean'] == '']
    kaggle = kaggle.drop(empty.index, axis=0)
    kaggle = kaggle.reset_index(drop=True)
    kaggle_bert = kaggle_bert.drop(empty.index, axis=0)
    kaggle_bert = kaggle_bert.reset_index(drop=True)

    # clean taglines
    kaggle['tagline'] = kaggle['tagline'].fillna('')
    kaggle = clean_kaggle(kaggle, 'tagline')
    kaggle_bert['tagline'] = kaggle_bert['tagline'].fillna('')
    kaggle_bert = clean_kaggle_bert(kaggle_bert, 'tagline')

    # clean titles
    kaggle['title'] = kaggle['title'].fillna('')
    kaggle = clean_kaggle(kaggle, 'title')
    kaggle_bert['title'] = kaggle_bert['title'].fillna('')
    kaggle_bert = clean_kaggle_bert(kaggle_bert, 'title')

    # remove taglines that are identical to plot
    same_as_plot = kaggle.loc[kaggle['tagline_clean'] == kaggle['plots_clean']]
    kaggle.loc[kaggle['tagline_clean'] == kaggle['plots_clean'], 'tagline_clean'] = ''
    for i in same_as_plot.index:
        kaggle_bert.loc[i, 'tagline_clean'] = ''

    # place tagline and title before plot
    kaggle['plots_new'] = [' '.join([title, tag, plot]) for title, tag, plot in
                           zip(kaggle['title_clean'], kaggle['tagline_clean'], kaggle['plots_clean'])]
    kaggle_bert['plots_new'] = [' '.join([title, tag, plot]) for title, tag, plot in
                                zip(kaggle_bert['title_clean'], kaggle_bert['tagline_clean'], kaggle_bert['plots_clean'])]

    # only keep genres and plots columns
    kaggle = kaggle[['genres', 'plots_new']]
    kaggle = kaggle.rename(columns={'plots_new': 'plots'})

    kaggle_bert = kaggle_bert[['genres', 'plots_new']]
    kaggle_bert = kaggle_bert.rename(columns={'plots_new': 'plots'})

    # remove incomprehensible or uninformative plots (all with less than 7 words)
    kaggle = kaggle[kaggle['plots'].str.split().str.len().ge(7)]
    kaggle_bert = kaggle_bert[kaggle_bert['plots'].str.split().str.len().ge(7)]

    # save cleaned data
    kaggle = kaggle.reset_index(drop=True)
    kaggle.to_pickle('./kaggle.pkl')
    print('Saved clean Kaggle to pickle.')

    kaggle_bert = kaggle_bert.reset_index(drop=True)
    kaggle_bert.to_pickle('./kaggle_BERT.pkl')
    print('Saved Kaggle for BERT to pickle.')

    # print statistics
    num_genres = number_genres(genre_list)
    print_kaggle_stats(kaggle, num_genres)
    print('For BERT:')
    print_kaggle_stats(kaggle_bert, num_genres)


def cmu_preprocessing(meta_path=CMU_ORIGINAL_META, plot_path=CMU_ORIGINAL_PLOT):
    #### LOAD DATA ####
    cmu_meta = pd.read_csv(meta_path, sep='\t', header=None)
    cmu_plot = pd.read_csv(plot_path, sep='\t', header=None)

    # rename columns
    cmu_meta.columns = ['wiki_id', 'fb_id', 'title', 'release', 'revenue', 'runtime', 'language', 'country', 'genres']
    cmu_plot.columns = ['wiki_id', 'plots']

    # merge title, genres and plot on wiki ID
    cmu = pd.merge(cmu_meta[['wiki_id', 'title', 'genres']], cmu_plot, on='wiki_id', sort=False)
    cmu = cmu.drop('wiki_id', axis=1)

    #### GENRE PROCESSING ####
    # transform genres to list, set them as new target column
    genre_list = genres_to_list_cmu(cmu)
    genres_lower = [[g.lower() for g in genres] for genres in genre_list]
    cmu['genres'] = genres_lower

    cmu = remove_empty_genres_cmu(cmu)

    #### PLOT PROCESSING ####
    # drop first duplicate, only then "keep=first" in remove_duplicates can be applied
    cmu = cmu.drop([2358], axis=0)
    cmu = remove_duplicates(cmu)

    # create copy of data because of different cleaning process for BERT
    cmu_bert = cmu.copy()
    # clean plots
    cmu = clean_cmu(cmu, 'plots')
    cmu_bert = clean_cmu_bert(cmu_bert, 'plots')

    # clean titles
    cmu = clean_cmu(cmu, 'title')
    cmu_bert = clean_cmu_bert(cmu_bert, 'title')

    # place tagline and title before plot
    cmu['plots_new'] = [' '.join([title, plot]) for title, plot in
                        zip(cmu['title_clean'], cmu['plots_clean'])]
    cmu_bert['plots_new'] = [' '.join([title, plot]) for title, plot in
                             zip(cmu_bert['title_clean'], cmu_bert['plots_clean'])]

    # only keep genres and plots
    cmu = cmu[['genres', 'plots_new']]
    cmu = cmu.rename(columns={'plots_new': 'plots'})

    cmu_bert = cmu_bert[['genres', 'plots_new']]
    cmu_bert = cmu_bert.rename(columns={'plots_new': 'plots'})

    # remove incomprehensible or uninformative plots (less than 9 or 13 words)
    cmu = cmu[cmu['plots'].str.split().str.len().ge(9)]
    cmu_bert = cmu_bert[cmu_bert['plots'].str.split().str.len().ge(13)]

    # save cleaned data
    cmu = cmu.reset_index(drop=True)
    cmu_bert = cmu_bert.reset_index(drop=True)
    print('Saved clean CMU to pickle.')

    cmu.to_pickle('./cmu.pkl')
    cmu_bert.to_pickle('./cmu_BERT.pkl')
    print('Saved CMU for BERT to pickle.')

    # print statistics
    num_genres = number_genres(genre_list)
    print_cmu_stats(cmu, num_genres)
    print('For BERT:')
    print_cmu_stats(cmu_bert, num_genres)


if __name__ == "__main__":
    # UNCOMMENT DEPENDING ON WHICH DATASET TO PROCESS
    kaggle_preprocessing()
    #cmu_preprocessing()
