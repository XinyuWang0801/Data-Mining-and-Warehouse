import helper
import sklearn

def fool_classifier(test_data):  ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...

    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance = helper.strategy()
    x_train = strategy_instance.class0 + strategy_instance.class1

    import numpy as np
    y = np.zeros((540, 1), dtype=np.int)
    y[360:] = 1
    y = y.ravel()  # Convert a multidimensional array to a one-dimensional array
    #print(y)
    #print('end of printing y')

    def createVocabList(dataSet):
        vocabSet = set([])
        for document in dataSet:
            vocabSet = vocabSet | set(document)
        return list(vocabSet)

    def setOfWords2Vec(vocabSet, inputSet):
        returnVec = [0] * len(vocabSet)
        for word in inputSet:
            if word in vocabSet:
                returnVec[vocabSet.index(word)] = 1
        return returnVec

    data = createVocabList(x_train)

    trainAll = []
    for postinDoc in x_train:
        trainAll.append(setOfWords2Vec(data, postinDoc))

    trainAll = np.array(trainAll)
    with open('test_data.txt', 'r') as test1:
        test1 = [line.strip().split(' ') for line in test1]

    testAll = []
    for postinDoc in test1:
        testAll.append(setOfWords2Vec(data, postinDoc))

    parameters = {}
    parameters['C'] = 0.05
    parameters['kernel'] = 'linear'
    parameters['degree'] = 3
    parameters['gamma'] = 1
    parameters['coef0'] = 1
    clf = strategy_instance.train_svm(parameters, trainAll, y)

    # -----------------
    #print('end of setting parameters')
    w = clf.coef_
    index = np.where(w[0] < 0)[0]

    dic_w = {}
    for i in index:
        dic_w[i] = w[0][i]
    dic_w = sorted(dic_w.items(), key=lambda d: d[1])[0:200]

    index = [dic_w[i][0] for i in range(len(dic_w))]

    add_word = []
    for i in index:
        add_word.append(data[i])

    n = 0
    for i in range(len(test1)):
        n = 0
        for w in add_word:
            if n == 20:
                break
            if w not in test1[i]:
                test1[i].append(w)
            else:
                continue
            n = n + 1

    #print('before open')
    file = open('./modified_data.txt', 'w')
    for i in range(len(test1)):
        file.write(" ".join(test1[i]))
        file.write("\n")
    file.close()

    ## You can check that the modified text is within the modification limits.
    modified_data = './modified_data.txt'
    #print('before assert')
    assert strategy_instance.check_data(test_data, modified_data)

'''
if __name__ == "__main__":
    test_data = './test_data.txt'
    strategy_instance = fool_classifier(test_data)
    #print('start showing results')


'''
