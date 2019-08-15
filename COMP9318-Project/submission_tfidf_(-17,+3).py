import helper
from sklearn.feature_extraction.text import TfidfTransformer,  TfidfVectorizer


def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...


    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance=helper.strategy()
    import numpy as np
    y = np.zeros((380,1),dtype=np.int)
    y[200:] = 1
    y = y.ravel()
    prng = np.random.RandomState(233233)
    train0 = prng.choice(strategy_instance.class0,200)
    train0 = train0.tolist()
    x_trainAll = train0 + strategy_instance.class1

    def createVocabList(dataSet):
        vocabSet=set([])
        for document in dataSet:
            vocabSet=vocabSet|set(document)
        return list(vocabSet)


    def setOfWords2Vec(vocabSet,inputSet):
        returnVec=[0]*len(vocabSet)
        for word in inputSet:
            if word in vocabSet:
                returnVec[vocabSet.index(word)]=1
        #print('type of returnVec',type(returnVec))
        return returnVec

    data = createVocabList(x_trainAll)
    trainAll=[]
    for postinDoc in x_trainAll :
        trainAll.append(setOfWords2Vec(data,postinDoc))


    trainAll = np.array(trainAll)
    with open('test_data.txt','r') as test1:           # test1 is the test_data file
            test1=[line.strip().split(' ') for line in test1]



    ####################### START OF TRANSFORM ############
    idf = TfidfTransformer()              # 类调用
    idf.fit(trainAll)
    xtrain_tfm = idf.transform(trainAll) # 将词频矩阵统计成TF-IDF值
    weight = xtrain_tfm.toarray()

    ####################### TEST PART #####################
    testAll=[]
    for postinDoc in test1:
        testAll.append(setOfWords2Vec(data,postinDoc))
    test = []
    test2 = []
    for i in range(len(test1)):
        test2.append(' '.join(test1[i]))
        test.append(' '.join(test1[i]))

    parameters={}
    parameters['C'] = 0.02
    parameters['kernel'] = 'linear'
    parameters['degree'] = 3
    parameters['gamma'] = 1
    parameters['coef0'] = 1
    clf = strategy_instance.train_svm(parameters,weight,y)
    w = clf.coef_

    index = np.where(w[0] > 0)[0]
    #print('show type of index: ',index)

    dic_w = {}
    for i in index:
        dic_w[i] = w[0][i]
    dic_w = sorted(dic_w.items(), key=lambda d: d[1],reverse = True)[0:-1]
    #dic_w_reverse = sorted(dic_w.items(), key=lambda d: d[1])[0:200]


    index = [dic_w[i][0] for i in range(len(dic_w))]
    #index_reverse = [dic_w_reverse[i][0] for i in range(len(dic_w))]

    delete_word = []
    for i in index:
        #print('word that need to be deleted: ',data[i])
        delete_word.append(data[i])
    #print('number of add_word: ',len(delete_word))

    with open('test_data.txt','r') as f:

            f=[line.strip().split(' ') for line in f]
            #print('len of set(f[0])',len(set(f[0])))
#删除单词
    n = 0
    lenth = 0
    for i in range(len(f)):         # 行数循环
        n = 0
        #print(i)
        deleted = set()
        added = set()
        for w in delete_word:          # 遍历要删除的数
            #print(w)
            if n == 20:
                break
            if w in test1[i]:
                for index,s in enumerate(f[i]):
                    #print(index,s)
                    if s == w:
                        test1[i].remove(s)
                        deleted.add(s)
                        n = n + 1
                        #删除一次'
            #print('number of deleting',len(deleted))
            if n < 20:
                for w in reversed(delete_word):
                    if n == 20:
                        break
                    if w not in test1[i]:
                        test1[i].append(w)
                        n += 1
                    else:
                        continue
            print('number of modified: ',n)
        lenth += 1
        n_reverse = 0
        '''   
        if len(deleted) != 20:
            print('len of set(deleted)',len(deleted))
            print('show set of deleted: ',deleted)
    print(len(set(f[0])))
    print(len(set(test1[0])))
        '''
    print('lenth',lenth)

    file=open('./modified_data.txt','w+')
    for i in range(len(test1)):
        file.write(" ".join(test1[i]))
        file.write("\n")
    file.close()



    ## You can check that the modified text is within the modification limits.
    modified_data='./modified_data.txt'
    if False:
        with open(modified_data, 'r') as mod:
            final_version = [line.strip().split(' ') for line in mod]
        data_final = createVocabList(final_version)
        final_ALL = []
        for postinDoc in final_version:
            final_ALL.append(setOfWords2Vec(data_final, postinDoc))

        final_ALL = np.array(final_ALL)




    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.

fool_classifier('./test_data.txt')

