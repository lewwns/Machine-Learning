from __future__ import division
import pylab
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, datasets
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.utils.np_utils import to_categorical
import csv
import pandas

#colnames = ['a', 'b', 'c', 'd']
#data1 = pandas.read_csv('rock_400.csv', names=colnames)
#data2 = pandas.read_csv('Jazz_400.csv', names=colnames)
#lyrics1 = data1.d.tolist()
#lyrics2 = data2.d.tolist()
#genre1 = data1.c.tolist()
#genre2 = data2.c.tolist()
#train_data = lyrics1[:300]+lyrics2[:300]
#test_data = lyrics1[300:]+lyrics2[300:]

#train_label = genre1[:300]+genre2[:300]
#test_label = genre1[300:]+genre2[300:]
#train_label_array = np.asarray(train_label)
#data = np.array(np.genfromtxt('country_rap_800.csv',delimiter=','))
#np.random.shuffle(data)
#X = np.reshape(data[:,3],(np.size(data[:,3]),1))
#y = np.reshape(data[:,2],(np.size(data[:,2]),1))

colnames = ['a', 'b', 'c', 'd']
data = pandas.read_csv('country_rap_800.csv', names=colnames)
#data2 = pandas.read_csv('rap_400.csv', names=colnames)
#data3 = pandas.read_csv('country_400.csv', names=colnames)
lyrics = data.d.tolist()
#lyrics2 = data2.d.tolist()
#lyrics3 = data3.d.tolist()
genre = data.c.tolist()
#genre2 = data2.c.tolist()
#genre3 = data3.c.tolist()
#lyrics = lyrics1+lyrics2
#genre = genre1+genre2
X = np.array(lyrics)
y = np.array(genre)
X = np.reshape(X,(len(lyrics),1))
y = np.reshape(y,(len(lyrics),1))

data = np.hstack((X,y))
#print len(lyrics)
#print len(genre)
#lst = []
#lst.append(lyrics)
#lst.append(genre)
#lst_array = np.asarray(lst)

#print len(lst_array)
#data = np.reshape(lst_array,(800,2))
#print data[:,1]
#data = np.zeros((len(genre),2))
#data = [[[]for i in xrange(2)] for j in xrange(len(genre))]
#data[:,0] = X[:,0]
#data[:,1] = y[:,0]
np.random.shuffle(data)
X = data[:,0]
y = data[:,1]
print y

#X = np.reshape(data[:,0],(np.size(data[:,0]),1))
#y = np.reshape(data[:,1],(np.size(data[:,1]),1))





#2 fold
kf = KFold(n_splits=10)
sum = 0
fold = 0
for train_index, test_index in kf.split(X):
    train_data, test_data = X[train_index], X[test_index]
    train_label, test_label = y[train_index], y[test_index]
    #training
    count_vec = CountVectorizer()
    X_train_counts = count_vec.fit_transform(train_data)
    transformer = TfidfTransformer()
    X_train_tfidf = transformer.fit_transform(X_train_counts)
    #clf= linear_model.LogisticRegression(C=1e5)
    #clf.fit(X_train_tfidf, train_label)
    ##SGD
 #   clf = linear_model.SGDClassifier()
#    clf.fit(X_train_tfidf, train_label)
    #testing
    X_new_counts = count_vec.transform(test_data)
    X_new_tfidf = transformer.transform(X_new_counts)
#    predicted = clf.predict(X_new_tfidf)

   # print('predict=',predicted)

       
    for i in range(0, len(train_label)):
        if train_label[i] == 'Rap':
            train_label[i] = 0
        elif train_label[i] == 'Country':
            train_label[i] = 1
        #else:
    #        train_label_array[i] = 2
            


    for i in range(0, len(test_label)):
        if test_label[i] == 'Rap':
            test_label[i] = 0
        elif test_label[i] == 'Country':
            test_label[i] = 1
       # else:
    #        test_label_array[i] = 2

    train_labels = to_categorical(train_label, num_classes=None)
    test_labels = to_categorical(test_label, num_classes=None)


    clf = Sequential()
    clf.add(Dense(64, input_dim=X_train_tfidf.shape[1], init='glorot_uniform', activation='tanh'))
    clf.add(Dropout(0.5))
    clf.add(Dense(64, activation='tanh'))
    clf.add(Dropout(0.5))
    clf.add(Dense(64, activation='tanh'))
    clf.add(Dropout(0.5))
    #clf.add(Dense(10, activation='tanh'))
    clf.add(Dense(2, activation='softmax'))
    rms = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    sgd = SGD(lr=0.001, decay = 1e-6, momentum=0.9, nesterov=True)
    clf.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    clf.fit(X_train_tfidf, train_labels, nb_epoch=20, batch_size=128) 
    score = clf.evaluate(X_new_tfidf, test_labels, batch_size=128)



    sum = sum+score[1]
    fold += 1

mean = float(sum)/fold
print mean


'''
train_data = lyrics1[:350]+lyrics2[:350]#+lyrics3[:350]
test_data = lyrics1[350:]+lyrics2[350:]#+lyrics3[350:]

train_label = genre1[:350]+genre2[:350]#+genre3[:350]
test_label = genre1[350:]+genre2[350:]#+genre3[350:]
train_label_array = np.asarray(train_label)

#print(train_label)
#print('test=',test_label)
#print('train=', train_label)
#count = 0
#for i in range(0, len(train_label)):
#    if train_label[i]=='Rap':
#        count += 1

#print('rap=',count)

#training
count_vec = CountVectorizer()
X_train_counts = count_vec.fit_transform(train_data)
transformer = TfidfTransformer()
X_train_tfidf = transformer.fit_transform(X_train_counts)
##KNN
#neigh = KNeighborsClassifier(n_neighbors=3)
#clf = neigh.fit(X_train_tfidf, train_label_array)
#NB
#X_train_tfidf = transformer.fit_transform(X_train_counts)
#clf = MultinomialNB().fit(X_train_tfidf, train_label_array)
##logistic
clf= linear_model.LogisticRegression(C=1e5)
clf.fit(X_train_tfidf, train_label_array)
#print (X_train_tfidf.shape)
#print (train_label_array.shape)
##perception
#clf = Perceptron(n_iter=50)
#clf.fit(X_train_tfidf, train_label_array)
##SGD
#clf = linear_model.SGDClassifier()
#clf.fit(X_train_tfidf, train_label_array)
##MLP
#clf = MLPClassifier(solver='lbfgs', activation='tanh',alpha=1e-3,early_stopping=True,max_iter=500,hidden_layer_sizes=(10,10,10), random_state=1)


#clf.fit(X_train_tfidf, train_label_array)
##LSTM
#model = Sequential()
#model.add(Dense(32, activation='relu', input_dim=100))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])

#model.fit(X_train_tfidf,train_label_array , epochs=10, batch_size=32)

#testing
X_new_counts = count_vec.transform(test_data)
X_new_tfidf = transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)

print('predict=',predicted)


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
