from __future__ import division
import pylab
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, datasets
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
#from keras.models import Sequential
#from keras.layers import Dense, Activation
import numpy as np
import csv
import pandas


colnames = ['a', 'b', 'c', 'd']
data = pandas.read_csv('con_ja_no.csv', names=colnames)
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

np.random.shuffle(data)
X = data[:,0]
y = data[:,1]

#2 fold
kf = KFold(n_splits=10)
sum = 0
fold = 0
all_words1 = {}
all_words2 = {}
all_words3 = {}
for train_index, test_index in kf.split(X):
    train_data, test_data = X[train_index], X[test_index]
    train_label, test_label = y[train_index], y[test_index]

    #training
    count_vec = CountVectorizer()
    X_train_counts = count_vec.fit_transform(train_data)
    transformer = TfidfTransformer()
    X_train_tfidf = transformer.fit_transform(X_train_counts)
    word = count_vec.get_feature_names()
    weight = X_train_tfidf.toarray()
    sum_weight = np.zeros((len(weight[1]),1))
    sum_weight = map(float, sum_weight)
    sum_weight2 = np.zeros((len(weight[1]),1))
    sum_weight2 = map(float, sum_weight2)
    sum_weight3 = np.zeros((len(weight[1]),1))
    sum_weight3 = map(float, sum_weight3)
    
    for i in range(len(weight)):
        if train_label[i] == 'Jazz':
            for j in range(len(word)):
                sum_weight[j] += weight[i][j]
        elif train_label[i] == 'Country':
            for j in range(len(word)):
                sum_weight2[j] += weight[i][j]
        #elif train_label[i] == 'Jazz':
#            for j in range(len(word)):
#                sum_weight3[j] += weight[i][j]
                

    country_w = OrderedDict()
    rap_w = OrderedDict()
    jazz_w = OrderedDict()
    scores = zip(word,sum_weight)
    scores2 = zip(word,sum_weight2)
    scores3 = zip(word,sum_weight3)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    sorted_scores2 = sorted(scores2, key=lambda x: x[1], reverse=True)
 #   sorted_scores3 = sorted(scores3, key=lambda x: x[1], reverse=True)

    i = 0
    #item[0]->word, item[1]->value
    for item in sorted_scores:
        if i<50:
            country_w[item[0]] = item[1]
            #print("{0:50} Score: {1}".format(item[0], item[1]))

            if all_words1.has_key(item[0]):
                all_words1[item[0]] += item[1]
            else:
                all_words1[item[0]] = item[1]               

            i+=1
    print "country end"
    k = 0
    for item in sorted_scores2:
        if k<50:
            rap_w[item[0]] = item[1]
            #print("{0:50} Score: {1}".format(item[0], item[1]))

            if all_words2.has_key(item[0]):
                all_words2[item[0]] += item[1]
            else:
                all_words2[item[0]] = item[1]
                
            k+=1
    print "rap end"
    '''   
    m = 0
    for item in sorted_scores3:
        if m<50:
            jazz_w[item[0]] = item[1]
            #print("{0:50} Score: {1}".format(item[0], item[1]))

            if all_words3.has_key(item[0]):
                all_words3[item[0]] += item[1]
            else:
                all_words3[item[0]] = item[1]
                
            m+=1
    print "jazz end"
    '''

    ##SGD
    clf = linear_model.LogisticRegression()
    clf.fit(X_train_tfidf, train_label)
    #testing
    X_new_counts = count_vec.transform(test_data)
    X_new_tfidf = transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)

   # print('predict=',predicted)

    count = 0
    for i in range(0, len(predicted)):
        if test_label[i] == predicted[i]:
            count += 1
        
        else:
            continue

    accuracy = float(count)/(len(predicted))
 #   print accuracy

    sum = sum+accuracy
    fold += 1

mean = float(sum)/fold
print mean


identical1 = {}
identical2 = {}
#identical3 = {}
print 'word1 never appeared in dict2,3 ********************'
for word in all_words1.keys():
    if all_words2.has_key(word): #and all_words3.has_key(word):
        identical1[word] = all_words1[word]
        identical2[word] = all_words2[word]
 #       identical3[word] = all_words3[word]
    elif not(all_words2.has_key(word)): #and not(all_words3.has_key(word)):
        print word, ',', all_words1[word]

print 'word2 never appeared in dict1,3***********************'
for word in all_words2.keys():
    if all_words1.has_key(word):# and all_words3.has_key(word):
        continue
    elif not(all_words1.has_key(word)):# and not(all_words3.has_key(word)):
        print word, ',', all_words2[word]
'''
print 'word3 never appeared in dict2,1***********************'
for word in all_words3.keys():
    if all_words2.has_key(word) and all_words1.has_key(word):
        continue
    elif not(all_words2.has_key(word)) and not(all_words1.has_key(word)):
        print word, ',', all_words3[word]
'''


print 'the identical word appeared in 1******************'
for word in identical1.keys():
    print word, ',', identical1[word]
print 'the identical word appeared in 2******************'
for word in identical2.keys():
    print word, ',', identical2[word]
'''
print 'the identical word appeared in 3******************'
for word in identical3.keys():
    print word, ',', identical3[word]
'''

'''
all1 = sorted(all_words1.items(), key=lambda item:item[1], reverse=True)
all2 = sorted(all_words2.items(), key=lambda item:item[1], reverse=True)
#all3 = sorted(all_words3.items(), key=lambda item:item[1], reverse=True)

print 'all_words1:'
for t in all1:
    print t[0], t[1]

print 'all_words2:'
for t in all2:
    print t[0], t[1]

#print 'all_words3:'
#for t in all3:
#    print t[0], t[1]

'''

    
#sum_sumlist_w = np.zeros((len(weight[1]),1))
#sum_sumlist_w2 = np.zeros((len(weight[1]),1))
#mean_sumlist_w = np.zeros((len(weight[1]),1))
#mean_sumlist_w2 = np.zeros((len(weight[1]),1))
#print 'sum=',sum_sumlist_w.shape

#for j in range(len(weight[1])):
#    for i in range(10):
#        sum_sumlist_w[j][0] += sum_wlist[i][j]
#        sum_sumlist_w2[j][0] += sum_wlist2[i][j]
    
#mean_sumlist_w = sum_sumlist_w/10
#mean_sumlist_w2 = sum_sumlist_w2/10
'''
country_w = OrderedDict()
rap_w = OrderedDict()
scores = zip(word,mean_sumlist_w)
scores2 = zip(word, mean_sumlist_w2)
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
sorted_scores2 = sorted(scores2, key=lambda x: x[1], reverse=True)
i = 0
for item in sorted_scores:
    if i<50:
        country_w[item[0]] = item[1]
        print("{0:50} Score: {1}".format(item[0], item[1]))
        i+=1
print "country end"
k = 0
for item in sorted_scores2:
    if k<50:
        rap_w[item[0]] = item[1]
        print("{0:50} Score: {1}".format(item[0], item[1]))
        k+=1
print "rap end"


    country_w = OrderedDict()
    rap_w = OrderedDict()
    scores = zip(word,sum_weight)
    scores2 = zip(word,sum_weight2)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    sorted_scores2 = sorted(scores2, key=lambda x: x[1], reverse=True)
    i = 0 
    for item in sorted_scores:
        if i<30:
            country_w[item[0]] = item[1]
            print("{0:50} Score: {1}".format(item[0], item[1]))
            i+=1
    print "country end"
    k = 0
    for item in sorted_scores2:
        if k<30:
            rap_w[item[0]] = item[1]
            print("{0:50} Score: {1}".format(item[0], item[1]))
            k+=1
    print "rap end"

            
 #   for k in range(len(word)):
#        print word[k], sum_weight[k]
       
    #clf= linear_model.LogisticRegression(C=1e5)
    #clf.fit(X_train_tfidf, train_label)
    ##SGD
    clf = linear_model.SGDClassifier()
    clf.fit(X_train_tfidf, train_label)
    #testing
    X_new_counts = count_vec.transform(test_data)
    X_new_tfidf = transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)

   # print('predict=',predicted)


    count = 0
    for i in range(0, len(predicted)):
        if test_label[i] == predicted[i]:
            count += 1
        
        else:
 #           print (test_label[i], predicted[i])
            continue

    accuracy = float(count)/(len(predicted))
    print accuracy

    sum = sum+accuracy
    fold += 1

mean = float(sum)/fold
print mean



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
