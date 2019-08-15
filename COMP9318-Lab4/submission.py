import pandas as pd
import numpy as np

        
def logistic_regression(data, labels, coefficients, num_epochs, learning_rate): # do not change the heading of the function
    import numpy as np
    def h(data, coefficients):      # hypothesis: matrix multiplication
        return 1 / (1 + np.exp(-1.0 * np.dot(data, coefficients)))

    data_list = data.tolist()
    print('data_list bofore insert: \n',data_list)
    for i in range(len(data_list)):
        data_list[i].insert(0,1)        # inser 1 into 0th position of list
    print('data_list:\n',data_list)
    data_np = np.array(data_list)
    print('data_np:',data_np)
    #print('labels_list:\n',labels_list)
    for i in range(num_epochs):
        hypothesis = h(data_np,coefficients)
        #print('H:',H)
        update = np.array([(labels - hypothesis) * data_np[:,j] for j in range(len(coefficients))]).transpose((1,0)) # matrix transpose
        updates = np.sum(update, axis = 0)
        #print('update',update)
        coefficients = coefficients + learning_rate * updates
    return coefficients


data_file='./asset/a'
raw_data = pd.read_csv(data_file, sep=',')
labels=raw_data['Label'].values
data=np.stack((raw_data['Col1'].values,raw_data['Col2'].values), axis=-1)

## Fixed Parameters. Please do not change values of these parameters...
coefficients = np.zeros(3) # We initialize the coefficients with ZERO. We also compute the intercept term.
num_epochs = 20000
learning_rate = 50e-5

coefficients = logistic_regression(data, labels, coefficients, num_epochs, learning_rate)
print(coefficients)