import helper
def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    
    
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance=helper.strategy() 
    parameters={}
    x_train = strategy_instance.class0 + strategy_instance.class1
    train = []
    for i in range(len(x_train)):
        train.append(' '.join(x_train[i]))
    
    with open(test_data,'r') as test1:
        test1=[line.strip().split(' ') for line in test1]
    
    
    test = []
    for i in range(len(test1)):
        test.append(' '.join(test1[i]))
    
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    tfv = CountVectorizer()
    tfv.fit(list(train))
    word = tfv.get_feature_names()

    xtrain_tfv = tfv.transform(train)
    xvalid_tfv = tfv.transform(test)
    print('len of word',len(word))
    
    import numpy as np
    y = np.zeros((540,1),dtype=np.int)
    y[360:] = 1
    y = y.ravel()
    
    from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
    parameters['C'] = 1.0
    parameters['kernel'] = 'linear'
    parameters['degree'] = 3
    parameters['gamma'] = 10
    parameters['coef0'] = 1
    clf = strategy_instance.train_svm(parameters,xtrain_tfv,y)

    predict_test = clf.predict(xvalid_tfv)
    count_1 = 0
    count_0 = 0
    for x in predict_test:
        if x == 1:
             count_1 += 1
        if x == 0:
             count_0 += 1
    print('count_1: ', count_1)
    print('count_0: ', count_0)
    #   完毕

    
    w = clf.coef_.toarray()
    index = np.where(w[0] < 0)[0]
    
    dic_w = {}
    for i in index:
        dic_w[i] = w[0][i]
    dic_w = sorted(dic_w.items(), key=lambda d: d[1])[0:100]
    
    index = [dic_w[i][0] for i in range(len(dic_w))]
    
    add_word = []
    for i in index:
        add_word.append(tfv.get_feature_names()[i])
    
    n = 0
    for i in range(len(test)):
        n = 0
        for w in add_word:
            if n == 20:
                break
            
            if w not in test1[i]:
                test[i] = test[i] + " " + w
                
            else:
                continue
            n = n + 1   
    
    file=open('./modified_data.txt','w')
    for i in range(len(test)):
        file.write(test[i])
        file.write("\n")
    file.close()
    
    
    ## You can check that the modified text is within the modification limits.
    modified_data='./modified_data.txt'
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.


fool_classifier('test_data.txt')