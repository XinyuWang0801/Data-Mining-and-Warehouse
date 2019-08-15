import helper
def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    
    
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance=helper.strategy() 
    import numpy as np
    y = np.zeros((360,1),dtype=np.int)
    y[180:] = 1
    y = y.ravel()
    prng = np.random.RandomState(233233)
    train0 = prng.choice(strategy_instance.class0,180)
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
        return returnVec
    
    data = createVocabList(x_trainAll)
    trainAll=[]
    for postinDoc in x_trainAll :
        trainAll.append(setOfWords2Vec(data,postinDoc))
    
    
    trainAll = np.array(trainAll)
    with open('test_data.txt','r') as test1:
            test1=[line.strip().split(' ') for line in test1]
    
    testAll=[]
    for postinDoc in test1:
        testAll.append(setOfWords2Vec(data,postinDoc))
    
    parameters={} 
    parameters['C'] = 0.02
    parameters['kernel'] = 'linear'
    parameters['degree'] = 3
    parameters['gamma'] = 1
    parameters['coef0'] = 1
    clf = strategy_instance.train_svm(parameters,trainAll,y)
    w = clf.coef_
    index = np.where(w[0] < 0)[0]
    
    dic_w = {}
    for i in index:
        dic_w[i] = w[0][i]
    dic_w = sorted(dic_w.items(), key=lambda d: d[1])[0:200]
    
    index = [dic_w[i][0] for i in range(len(dic_w))]

    for i in index:
        print('index: ',i)
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

    print(len(test1))
    file=open('./modified_data.txt','w+')
    for i in range(len(test1)):
        file.write(" ".join(test1[i]))
        file.write("\n")
    line_num = 0
    file.close()
    with open('./modified_data.txt') as mod:
        for line in mod:
            line_num += 1
    print('line num of modified_data',line_num)




    
    ## You can check that the modified text is within the modification limits.
    modified_data='./modified_data.txt'
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.


fool_classifier('./test_data.txt')