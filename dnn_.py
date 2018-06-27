from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, datasets
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.utils.np_utils import to_categorical
import numpy as np
import csv
import pandas


colnames = ['a', 'b', 'c', 'd']
data = pandas.read_csv('country_rap_jazz_1200_v2.csv', names=colnames)
#data2 = pandas.read_csv('rap_400.csv', names=colnames)
#data3 = pandas.read_csv('country_400.csv', names=colnames)
lyrics = data.d.tolist()
#lyrics2 = data2.d.tolist()
#lyrics3 = data3.d.tolist()
genre = data.c.tolist()
#genre2 = data2.c.tolist()
#genre3 = data3.c.tolist()
X = np.array(lyrics)
y = np.array(genre)
X = np.reshape(X,(len(lyrics),1))
y = np.reshape(y,(len(lyrics),1))

data = np.hstack((X,y))

np.random.shuffle(data)
X = data[:,0]
y = data[:,1]

kf = KFold(n_splits=10)
SUM = 0
fold = 0
for train_index, test_index in kf.split(X):
    train_data, test_data = X[train_index], X[test_index]
    train_label, test_label = y[train_index], y[test_index]


    train_label_array = np.array(train_label)
    test_label_array = np.asarray(test_label)

    #training
    count_vec = CountVectorizer()
    X_train_counts = count_vec.fit_transform(train_data)
    transformer = TfidfTransformer()
    X_train_tfidf = transformer.fit_transform(X_train_counts)

    # dim
    train_TFIDF = X_train_tfidf.toarray()
    dim = len(train_TFIDF[0])
    
    #DNN
    #testing
    X_new_counts = count_vec.transform(test_data)
    X_new_tfidf = transformer.transform(X_new_counts)

    for i in range(0, len(train_label_array)):
        if train_label_array[i] == 'Country':
            train_label_array[i] = 0
        elif train_label_array[i] == 'Jazz':
            train_label_array[i] = 1
        else:
            train_label_array[i] = 2
            


    for i in range(0, len(test_label_array)):
        if test_label_array[i] == 'Country':
            test_label_array[i] = 0
        elif test_label_array[i] == 'Jazz':
            test_label_array[i] = 1
        else:
            test_label_array[i] = 2

    train_labels = to_categorical(train_label_array, num_classes=None)
    test_labels = to_categorical(test_label_array, num_classes=None)



                    
    #good!!
    clf = Sequential()
    clf.add(Dense(64, input_dim=dim, init='glorot_uniform', activation='tanh'))
    clf.add(Dropout(0.5))
    clf.add(Dense(64, activation='tanh'))
    clf.add(Dropout(0.5))
    clf.add(Dense(64, activation='tanh'))
    clf.add(Dropout(0.5))
    #clf.add(Dense(10, activation='tanh'))
    clf.add(Dense(3, activation='softmax'))
    rms = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    sgd = SGD(lr=0.001, decay = 1e-6, momentum=0.9, nesterov=True)
    clf.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    clf.fit(X_train_tfidf, train_labels, nb_epoch=20, batch_size=128) 
    score = clf.evaluate(X_new_tfidf, test_labels, batch_size=128)

    print 'score', score
    SUM += score[1]
    fold += 1

mean = float(SUM)/fold
print 'mean', mean

'''		
#good!!
clf = Sequential()
clf.add(Dense(64, input_dim=16772, init='glorot_uniform', activation='tanh'))
clf.add(Dropout(0.5))
clf.add(Dense(64, activation='tanh'))
clf.add(Dropout(0.5))
clf.add(Dense(64, activation='tanh'))
clf.add(Dropout(0.5))
#clf.add(Dense(10, activation='tanh'))
clf.add(Dense(3, activation='softmax'))
rms = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = SGD(lr=0.001, decay = 1e-6, momentum=0.9, nesterov=True)
clf.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
clf.fit(X_train_tfidf, train_labels, nb_epoch=20, batch_size=128) 
score = clf.evaluate(X_new_tfidf, test_labels, batch_size=128)

print score
'''

'''
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=15783))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

for i in range(0, len(train_label_array)):
	if train_label_array[i] == 'Jazz':
		train_label_array[i] = 0
	elif train_label_array[i] == 'Rap':
		train_label_array[i] = 1
	else:
		train_label_array[i] = 2
		
model.fit(X_train_tfidf,train_label_array , epochs=10, batch_size=128)

#testing
X_new_counts = count_vec.transform(test_data)
X_new_tfidf = transformer.transform(X_new_counts)

for i in range(0, len(test_label_array)):
	if test_label_array[i] == 'Jazz':
		test_label_array[i] = 0
	elif test_label_array[i] == 'Rap':
		test_label_array[i] = 1
	else:
		test_label_array[i] = 2

score = model.evaluate(X_new_tfidf, test_label_array, batch_size=128)
print 'score=', score
'''
'''
predicted = clf.predict(X_new_tfidf)

print('predict=',predicted)

for i in range(0, len(test_label)):
	if test_label[i] == 'Jazz':
		test_label[i] = 0
	elif test_label[i] == 'Rap':
		test_label[i] = 1
	else:
		test_label[i] = 2


count = 0
for i in range(0, len(predicted)):
    if test_label[i] == predicted[i]:
        count += 1
        
    else:
        print (test_label[i], predicted[i])
        continue

accuracy = float(count)/(len(predicted))
print accuracy
'''
