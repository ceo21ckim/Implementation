from __future__ import print_function 
import time
import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split

from slir import slir

def main():
    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()

    # Use only one feature 
    diabetes_x = diabetes.data 

    # Split the data into training/testing sets 

    # diabetes_x_train, diabetes_x_test, diabetes_y_train, diabetes_y_test = train_test_split(diabetes_x, diabetes.target, train_size = 0.9)
    diabetes_x_train = diabetes_x[:-20]
    diabetes_x_test = diabetes_x[-20:]
    
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]
    
    diabetes_x_train = np.hstack([diabetes_x_train for _ in range(43)])
    diabetes_x_test = np.hstack([diabetes_x_test for _ in range(43)])
    
    print("start regression")
    
    start_time = time.time()
    clf = slir.SparseLinearRegression(n_iter=200, verbose=True, verbose_skip=5)
    
    # fit 
    clf.fit(diabetes_x_train, diabetes_y_train)
    print("Processing Time: %.4f"% (time.time() - start_time))
    
    # Predict 
    predicted_labels = clf.predict(diabetes_x_test)
    
    # Scores of correlation and mean squared error(MSE)
    print('Correlation: %.4f' % np.corrcoef(predicted_labels, diabetes_y_test)[0, 1])
    print("MSE : %.4f" % np.mean((predicted_labels - diabetes_y_test) ** 2))
    
if __name__ == "__main__":
    main()