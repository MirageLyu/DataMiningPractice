import pkuseg
import numpy as np
import pandas as pd
import pickle
import gc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import lightgbm as lgb
import tensorflow as tf
from tensorflow.contrib import rnn
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import jieba
from sklearn.ensemble import BaggingClassifier

diminput = 50
dimhidden = 128
dimoutput = 72
nsteps = 50

def RNN(X, W, b, nsteps):
    X = tf.transpose(X, [1,0,2])
    X = tf.reshape(X, [-1, diminput])
    H_1 = tf.matmul(X, W["h1"]) + b["b1"]
    H_1 = tf.split(H_1, nsteps, 0)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(dimhidden, forget_bias=1.0)
    LSTM_O, LSTM_S = rnn.static_rnn(lstm_cell, H_1, dtype=tf.float32)
    O = tf.matmul(LSTM_O[-1], W["h2"]) + b["b2"]
    print("Network ready.")
    return  {"X":X,"H_1":H_1,"LSTM_O":LSTM_O,"LSTM_S":LSTM_S,"O":O}

def myRNN(words_vec, Solutions, test_data):
    W = {"h1" : tf.Variable(tf.random_normal([diminput, dimhidden])),
         "h2" : tf.Variable(tf.random_normal([dimhidden, dimoutput]))}
    b = {"b1" : tf.Variable(tf.random_normal([dimhidden])),
         "b2" : tf.Variable(tf.random_normal([dimoutput]))}

    learning_rate = 0.001
    x = tf.placeholder("float", [None, nsteps, diminput])
    y = tf.placeholder("float", [None, dimoutput])
    myrnn = RNN(x, W, b, nsteps)
    pred = myrnn['O']
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    optm = tf.train.GradientDescentOptimizer(learning_rate).minimize((cost))
    accr = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))
    init = tf.global_variables_initializer()
    print("Netword Ready!")

    training_epochs = 5
    batch_size = 32
    display_step = 1
    sess = tf.Session()
    sess.run(init)
    print("Start optimization")
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(len(Solutions)/batch_size)

        p = 0
        q = min(len(Solutions), batch_size)

        while p < len(Solutions):
            batch_xs = words_vec[p:q]
            batch_ys = Solutions[p:q]
            feeds = {x : batch_xs, y : batch_ys}
            sess.run(optm, feed_dict=feeds)

            avg_cost += sess.run(cost, feed_dict=feeds)/total_batch
            p = q
            q  = min(q+batch_size, len(Solutions))
        if epoch % display_step == 0:
            print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
            feeds = {x: batch_xs, y:batch_ys}

def Text_Cut_Thulac_v():
    #增加标词性
    pku1 = pkuseg.pkuseg(model_name='web', postag=True)

    dataPath = 'Mining_Challenge_Dataset/train.data'
    outPath = 'Mining_Challenge_Dataset/train_cut_v.data'

    file = open(dataPath, 'r')
    outfile = open(outPath, 'w')
    texts = file.readlines()

    texts_len = len(texts)

    for i in range(texts_len):
        texts[i] = pku1.cut(texts[i])
        if i % 1000 == 0:
            percent = float(i) / float(texts_len)
            print('Cutting: ' + str(round(percent, 4)))
        for word in texts[i]:
            print(word, file=outfile, end=' ')
        print('', file=outfile)

def Test_Text_Cut_Thulac_v():
    #增加标词性
    pku1 = pkuseg.pkuseg(model_name='web', postag=True)

    dataPath = 'Mining_Challenge_Dataset/test.data'
    outPath = 'Mining_Challenge_Dataset/test_cut_v.data'

    file = open(dataPath, 'r')
    outfile = open(outPath, 'w')
    texts = file.readlines()

    texts_len = len(texts)

    for i in range(texts_len):
        texts[i] = pku1.cut(texts[i])
        if i % 1000 == 0:
            percent = float(i) / float(texts_len)
            print('Cutting: ' + str(round(percent, 4)))
        for word in texts[i]:
            print(word, file=outfile, end=' ')
        print('', file=outfile)

def Text_Cut_Thulac():
    #不增加标词性
    dataPath = 'Mining_Challenge_Dataset/train.data'
    outPath = 'Mining_Challenge_Dataset/train_cut.data'

    file = open(dataPath, 'r', encoding ="utf-8-sig")
    outfile = open(outPath, 'w', encoding="utf-8-sig")
    texts = file.read().splitlines()

    texts_len = len(texts)

    for i in range(texts_len):
        text = jieba.cut(texts[i], cut_all=False)
        texts[i] = " ".join(text)
        if i % 1000 == 0:
            percent = float(i) / float(texts_len)
            print('Cutting: ' + str(round(percent, 4)))
        print(texts[i], file=outfile)

    file.close()
    outfile.close()

def Test_Text_Cut_Thulac():
    #不增加标词性
    dataPath = 'Mining_Challenge_Dataset/test.data'
    outPath = 'Mining_Challenge_Dataset/test_cut.data'

    file = open(dataPath, 'r', encoding ="utf-8-sig")
    outfile = open(outPath, 'w', encoding ="utf-8-sig")
    texts = file.read().splitlines()

    texts_len = len(texts)

    for i in range(texts_len):
        texts[i] = texts[i].split('\t')[1]
        text = jieba.cut(texts[i], cut_all=False)
        texts[i] = " ".join(text)
        if i % 1000 == 0:
            percent = float(i) / float(texts_len)
            print('Cutting: ' + str(round(percent, 4)))
        print(texts[i], file=outfile)

    file.close()
    outfile.close()

def AntiStopWords():
    outPath = 'Mining_Challenge_Dataset/train_cut_stopped.data'
    outfile = open(outPath, 'w', encoding ='utf-8-sig')

    stop_words = None
    with open("Mining_Challenge_Dataset/stopwords.data", "r", encoding ='utf-8-sig') as f:
        stop_words = f.readlines()
        stop_words = [word.replace("\n", "") for word in stop_words]

    dataPath = 'Mining_Challenge_Dataset/train_cut.data'

    file = open(dataPath, 'r', encoding='utf-8-sig')
    texts = file.readlines()

    for i, line in enumerate(texts):
        for word in stop_words:
            if word in line:
                line = line.replace(word, "")
        texts[i] = line

    outfile.writelines(texts)
    f.close()
    outfile.close()
    file.close()

def Test_AntiStopWords():
    outPath = 'Mining_Challenge_Dataset/test_cut_stopped.data'
    outfile = open(outPath, 'w', encoding='utf-8-sig')

    stop_words = None
    with open("Mining_Challenge_Dataset/stopwords.data", "r", encoding='utf-8-sig') as f:
        stop_words = f.readlines()
        stop_words = [word.replace("\n", "") for word in stop_words]

    dataPath = 'Mining_Challenge_Dataset/test_cut.data'

    file = open(dataPath, 'r', encoding='utf-8-sig')
    texts = file.readlines()

    for i, line in enumerate(texts):

        for word in stop_words:
            if word in line:
                line = line.replace(word, "")
        texts[i] = line

    outfile.writelines(texts)
    outfile.close()
    file.close()
    f.close()

def Count_Vectorizer():
    dataPath = 'Mining_Challenge_Dataset/train_cut.data'
    file = open(dataPath, 'r', encoding ='utf-8-sig')
    texts = file.read().splitlines()

    testPath = 'Mining_Challenge_Dataset/test_cut.data'
    testfile = open(testPath, 'r', encoding='utf-8-sig')
    testtexts = testfile.read().splitlines()

    train_data = texts + testtexts

    count_vect = CountVectorizer()
    result = count_vect.fit_transform(train_data)

    word_vec = result[0:len(texts)]
    test_vec = result[len(texts):len(train_data)]

    print("CountVectorizer Model Successfully.")

    return word_vec, test_vec

def TFIDF_Vectorizer_Generate_Model_Pickle():
    dataPath = 'Mining_Challenge_Dataset/train_cut_stopped.data'

    file = open(dataPath, 'r')
    texts = file.read().splitlines()

    testPath = 'Mining_Challenge_Dataset/test_cut_stopped.data'

    testfile = open(testPath, 'r')
    testtexts = testfile.read().splitlines()

    texts = texts + testtexts

    count_vect = TfidfVectorizer()
    count_vect.fit(texts)

    pickle.dump(count_vect, open('vector.model', 'wb'))
    print("TFIDF_Vectorizer Model Dumped Successfully.")

def w2v_Mean_Vectorizer():
    dataPath = 'Mining_Challenge_Dataset/train_cut.data'
    testPath = 'Mining_Challenge_Dataset/test_cut.data'


    file = open(dataPath, 'r', encoding="utf-8-sig")
    texts = file.read().splitlines()
    test_file = open(testPath, 'r', encoding="utf-8-sig")
    testtexts = test_file.read().splitlines()

    train_data = texts + testtexts

    for i in range(len(train_data)):
        train_data[i] = train_data[i].split(' ')
        train_data[i] = [word for word in train_data[i] if word != '']
                                                                                                                                                                    

    print("Training Doc2Vec")
    vect = Word2Vec(train_data, size=300, min_count=1, sg=1, workers=4)
    #vect = Word2Vec.load("w2vdoc2vec.model")

    print("Saving w2vDoc2Vec Model")
    vect.save("w2vdoc2vec.model")
    print("Model Saved.")

    words_vec = np.zeros((len(texts), 300), dtype=np.float)
    test_vec = np.zeros((len(testtexts), 300), dtype=np.float)

    for i in range(len(testtexts)):
        test_text = train_data[i + len(texts)]

        for word in test_text:
            test_vec[i] = np.add(test_vec[i], vect[word])

        test_vec[i] = np.divide(test_vec[i], len(test_text))

        if i % 1000 == 0:
            print("Testing data vectoring: " + str(float(i) / float(len(testtexts))))

    for i in range(len(texts)):
        test_text = train_data[i]

        for word in test_text:
            words_vec[i] = np.add(words_vec[i], vect[word])

        words_vec[i] = np.divide(words_vec[i], len(test_text))

        if i%1000 == 0:
            print("Training data vectoring: " + str(float(i)/float(len(texts))))


    file.close()
    test_file.close()

    return words_vec, test_vec

def Vectorizer():
    dataPath = 'Mining_Challenge_Dataset/train_cut.data'
    testPath = 'Mining_Challenge_Dataset/test_cut.data'


    file = open(dataPath, 'r', encoding="utf-8-sig")
    texts = file.read().splitlines()
    test_file = open(testPath, 'r', encoding="utf-8-sig")
    testtexts = test_file.read().splitlines()

    train_data = texts + testtexts

    train_data_tg = []

    for i, text in enumerate(train_data):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l-1] = word_list[l-1].strip()
        word_list = [word for word in word_list if word != '' and word != '，' and word != '。']
        document = TaggedDocument(word_list, tags=[i])
        train_data_tg.append(document)
    #count_vect = pickle.load(open('vector.model', 'rb'))


    print("Training Doc2Vec")
    vect = Doc2Vec(train_data_tg, workers=4, min_count=1, vector_size=50)
    #vect = Doc2Vec.load("doc2vec.model")

    del train_data
    del train_data_tg
    gc.collect()

    print("Saving Doc2Vec Model")
    vect.save("doc2vec.model")
    print("Model Saved.")

    words_vec = np.zeros((len(texts), 50))
    test_vec = np.zeros((len(testtexts), 50))

    for i in range(len(texts)):
        test_text = texts[i].split(' ')
        test_text = [word for word in test_text if word != '' and word != '，' and word != '。']
        words_vec[i] = vect.infer_vector(doc_words=test_text)
        if i%1000 == 0:
            print("Training data vectoring: " + str(float(i)/float(len(texts))))

    for i in range(len(testtexts)):
        test_text = testtexts[i].split(' ')
        test_text = [word for word in test_text if word != '' and word != '，' and word != '。']
        test_vec[i] = vect.infer_vector(doc_words=test_text)
        if i%1000 == 0:
            print("Testing data vectoring: " + str(float(i)/float(len(testtexts))))

    file.close()
    test_file.close()

    return words_vec, test_vec

    #while q < len(texts):
    #    tmp_text = texts[p:q]
    #    # 此处应修改为 分片转换(无影响，因为模型已训练)
    #    words_vec = count_vect.transform(tmp_text)
    #    word_list = words_vec.toarray().tolist()

    #    p = q
    #    q = max(q+step, len(texts))

def readSolutionMap():
    filePath = 'Mining_Challenge_Dataset/emoji.data'
    emojifile = open(filePath, 'r', encoding='utf-8-sig')

    SolutionMap = {}

    index = 0

    lines = emojifile.read().splitlines()
    for line in lines:
        key = index
        index += 1
        value = line.split('\t')[1]
        SolutionMap[value] = key
    return SolutionMap

def transformSolutionToIndex(sol_map):
    filePath = 'Mining_Challenge_Dataset/train.solution'
    solution = open(filePath, 'r', encoding='utf-8-sig')

    lines = solution.readlines()

    result_list = [0 for i in range(len(lines))]

    index = 0

    for line in lines:
        key = line.split('{')[1].split('}')[0]
        result_list[index] = sol_map[key]
        index += 1

    return result_list

def Naive_Bayes(words_vec, Solutions, test_data):
    clf = MultinomialNB()
    clf.fit(words_vec, Solutions)
    result = clf.predict(test_data)

    outfile = open('Output/nbtest.csv', 'w')
    outfile.write('ID,Expected\n')

    index = 0

    for item in result:
        outfile.write(str(index) + ',' + str(item) + '\n')
        index += 1
    outfile.close()

def MLP(words_vec, Solutions, test_data):
    print("Start training MLP.")

    clf = MLPClassifier(solver = 'adam', hidden_layer_sizes=(60, 60))
    #clf.fit(words_vec, Solutions)
    #model_file = open("mlp.model", "wb")
    #pickle.dump(clf, file = model_file)
    #model_file.close()

    bagging = BaggingClassifier(clf, max_samples=0.5, max_features=0.8)
    bagging.fit(words_vec, Solutions)

    #model_file = open("mlp.model", "rb")
    #clf = pickle.load(model_file)
    #model_file.close()

    print("Model Trained Finished.")



    #result = clf.predict(test_data)
    result= bagging.predict(test_data)

    print("Predict Finished.")

    outfile = open('Output/mlptest.csv', 'w')
    outfile.write('ID,Expected\n')

    print("Output to file successfully.")

    index = 0

    for item in result:
        outfile.write(str(index) + ',' + str(item) + '\n')
        index += 1
    outfile.close()

# SVM GG
def SVM(words_vec, Solutions, test_data):
    clf = svm.SVC(C=0.8, kernel='rbf', gamma='auto', decision_function_shape='ovr')
    clf.fit(words_vec, Solutions)
    result = clf.predict(test_data)

    outfile = open('Output/svm_test.csv', 'w')
    outfile.write('ID,Expected\n')

    index = 0

    for item in result:
        outfile.write(str(index) + ',' + str(item) + '\n')
        index += 1
    outfile.close()

def LightGBM(words_vec, Solutions, test_data):
    train_data = lgb.Dataset(words_vec.astype(np.float), label=Solutions)
    param = {'objective': 'multiclass', 'boosting': 'gbdt', 'num_class': 72, 'device_type': 'cpu'}

    num_round = 20
    bst = lgb.train(param, train_data, num_boost_round=num_round)

    bst.save_model('LightGBM.model')

    result = bst.predict(test_data.astype(np.float))

    outfile = open('Output/LightGBM_test.csv', 'wTeri')
    outfile.write('ID,Expected\n')

    index = 0

    result=[list(x).index(max(x)) for x in result]

    for item in result:
        outfile.write(str(index) + ',' + str(item) + '\n')
        index += 1
    outfile.close()


if __name__ == '__main__':
    #Text_Cut_Thulac()
    #AntiStopWords()

    #Test_Text_Cut_Thulac()
    #Test_AntiStopWords()

    #train, test = Count_Vectorizer()

    train, test = w2v_Mean_Vectorizer()

    #Naive_Bayes(train, transformSolutionToIndex(readSolutionMap()), test)

    MLP(train, transformSolutionToIndex(readSolutionMap()), test)


