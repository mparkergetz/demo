### knn_fxns.py
import numpy as np
import pandas as pd

def row_distance(row1, row2):
    '''
    Returns the distance between two input rows, extracted from a Pandas dataframe
    INPUT: two rows which should be Pandas series or array-type, not data frame 
    OUTPUT: Euclidean distance
    '''
    array1 = np.array(row1)
    array2 = np.array(row2)
    dist = np.sqrt(sum((array1-array2)**2))
    return dist

def calc_distance_to_all_rows(df,example_row):
    '''
    Computes distance between every row in input df (Pandas dataframe) and example_row (Pandas series or array type)
    Calls 'row_distance'
    INPUT: df, Pandas dataframe; example_row
    OUTPUT:Pandas dataframe with additional column 'distance_to_ex' added to input dataframe df
    '''
    distance_list = []
    no_class_df = df.drop(['class'],axis=1)
    num_rows = len(no_class_df)
    for index in range(0,num_rows):
        current_row = no_class_df.iloc[index,:]
        distance = row_distance(current_row, example_row)
        distance_list.append(distance)
    return_df = df.assign(distance_to_ex=distance_list)
    return return_df
  

def find_k_closest(df, example_row, k):
    """
    Finds the k closest neighbors to example, excluding the example itself.
    Calls 'calc_distance_to_all_rows'
    IF there is a tie for kth closest, choose the final k to include via random choice.
    INPUT: df, Pandas dataframe; example_row, Pandas series or array type; k, integer number of nearest neighbors.
    OUTPUT: dataframe in same format as input df but with k rows and sorted by 'distance_to_ex.'
    """
    dist_df = calc_distance_to_all_rows(df,example_row)
    #print(dist_df)
    sort_df = dist_df.sort_values(by=['distance_to_ex'])
    drop_head_df = sort_df.drop(sort_df.index[0])
    if (sort_df.index[k] == sort_df.index[k+1]):
        choice_list = [0,1]
        choice = np.random.choice(choice_list,1)
        if choice==1:
            drop_head_df.drop(drop_head_df.index[k])
    ret_df = drop_head_df.head(k)       
    return ret_df
    
    
def classify(df, example_row, k):
    """
    Return the majority class from the k nearest neighbors of example
    Calls 'find_k_closest'
    INPUT: df, Pandas dataframe; example_row, Pandas series or array type; k, integer number of nearest neighbors
    OUTPUT: string referring to closest class.
    """
    ret_df = find_k_closest(df,example_row,k)
    most = ret_df['class'].max()
    return most
    

def evaluate_accuracy(training_df, test_df,k):
    no_class_df = test_df.drop(['class'],axis=1)
    num_correct = 0    
    num_rows = len(test_df)
    for row_num in range(num_rows):
        test_row = no_class_df.iloc[row_num,:]
        if classify(training_df,test_row,k) == test_df['class'].iloc[row_num]:
            num_correct += 1
    return num_correct/num_rows

