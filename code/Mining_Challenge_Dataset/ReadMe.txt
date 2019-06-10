Welcome to the MINING CHALLENGE for NJU Introduction to Data Mining 2019! Your target is to recommend suitable emojis for messages. Here comes a brief introduction to the data files.
train.data -- your training data; 863,164 lines in total; each message in a line
train.solution -- the ground truth of training data; 863,164 lines in total either; messages and emojis are corresponding in order
test.data -- your test data; each line includes a number which means the index of the test message and a test message, with a tab character between them
emoji.data -- the dataset has 72 emojis in total; we build a one-to-one correspondence between all 72 emojis and integers from 0 to 71
sample_submission.csv -- your submission in Kaggle should be in a csv file; the csv file has two columns, the first column is the index of the test data which should be ranked in ascending order with no index missing and the second column is the index of the emoji according to your prediction; the first row is fixed of 'ID' and 'Expected', corresponding to each column
Please feel free to contact TAs if you have any question.
GOOD LUCK!
