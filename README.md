# Multi-Label Classification of Movie Genres by Plot Summaries

The two datasets used are [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) from Kaggle 
and the [CMU Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/).


- preprocessing.py: cleans original data, saves new files, prints dataset statistics

- gridsearch.py: prints results of specified grid search

- baseline.py: trains and tests specified baseline classifier, prints test results

- inference.py: prints predicted genres for movie plots included in inference_input_text.txt, 
	        uses specified baseline classifier depending on whether kaggle or cmu was chosen

- [bilstm, bert]_run.py: trains and tests bilstm or bert, saves metrics as .pkl, saves metrics for TensorBoard in new runs directory 

- [bilstm, bert]_data.py: used by [bilstm, bert]_run.py to prepare data

- [bilstm, bert]_model.py: used by [bilstm, bert]_run.py to get the model, training and evaluation functions 

- utils.py: used by almost all other classes to get functions for pre-processing, cleaning, statistics, splitting



To run bilstm_run.py or use the *infer_from_bilstm* function in inference.py, Word2Vec is necessary.
Download the file "GoogleNews-vectors-negative300.bin.gz" [here](https://code.google.com/archive/p/word2vec/), 
and place it in the same directory as the .py-files.








 
	






aldj