# Sentiment-Analysis-of-Movie-Review

## Description
To analyze movie's review and its sentiment and do classification of new review either it is positive or negative using deep learning and LSTM

* Model training - Deep learning
* Method: LSTM
* Visualization toolkit: Tensorboard

In this analysis, dataset used from https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv.

### About The Dataset:

It has 50,000 dataset with 2 columns, Review and Sentiment
Review is in a series of string written by movie reviewer
Sentiment, either positive or negative is the sentiment set by a human based on the movie review

<p align="center">
  <img width="360" src="https://github.com/snaffisah/Sentiment-Analysis-of-Movie-Review/blob/main/Static/dataset.JPG">
</p>

### Tensorboard
In this analysis, the visualizing metrics such as loss and accuracy will be appear and keep track on Tensorboard

To open tensorboard, 
1. Open command prompt
2. Activate environment (example: conda activate tf_env)
3. Run tensorboard --logdir "PATH of log file"
4. If no error, you may open http://localhost:6006/ on browser

<p align="center">
  <img src="https://github.com/snaffisah/Sentiment-Analysis-of-Movie-Review/blob/main/Static/tensorboard_bind.JPG">
</p>

Loss & accuracy scalar:
<p align="center">
  <img src="https://github.com/snaffisah/Sentiment-Analysis-of-Movie-Review/blob/main/Static/loss%20and%20accuracy%20TB.JPG">
</p>

Workflow graph:
<p align="center">
  <img width="460" src="https://github.com/snaffisah/Sentiment-Analysis-of-Movie-Review/blob/main/Static/20220511-114903_train.png">
</p>

### How to test model

1. Clone this repository
2. Run python file --> deploy.py
3. Console will prompt user to input new review
4. Press 'Enter'
5. Result of sentiment will auto appear below inserted review

Positive sentiment review:
<p align="center">
  <img src="https://github.com/snaffisah/Sentiment-Analysis-of-Movie-Review/blob/main/Static/positive%20review.JPG">
</p>

Negative sentiment review:
<p align="center">
  <img src="https://github.com/snaffisah/Sentiment-Analysis-of-Movie-Review/blob/main/Static/negative%20review.JPG">
</p>

Enjoy!
