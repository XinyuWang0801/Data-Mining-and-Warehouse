## import modules here 
import pandas as pd
import numpy as np
import copy


################# Question 1 #################

# helper functions
def project_data(df, d):
    # Return only the d-th column of INPUT
    return df.iloc[:, d]

def select_data(df, d, val):
    # SELECT * FROM INPUT WHERE input.d = val
    col_name = df.columns[d]
    return df[df[col_name] == val]

def remove_first_dim(df):
    # Remove the first dim of the input
    return df.iloc[:, 1:]

def slice_data_dim0(df, v):
    # syntactic sugar to get R_{ALL} in a less verbose way
    df_temp = select_data(df, 0, v)
    return remove_first_dim(df_temp)



def one_dim_duc(df):
    vals = list(df.loc[0])
    result = [vals[:-1]]
    for i, val in enumerate(result):
        temp = deepcopy(val)
        for j, list_val in enumerate(temp):
            if list_val != 'ALL':
                temp2 = deepcopy(temp)
                temp2[j] = 'ALL'
                if temp2 not in result:
                    result.append(temp2)

    for i in result:
        i.append(vals[-1])
    result = pd.DataFrame(result, columns=list(df))
    return result

def _buc_rec_optimized(df, pre_num, df_out):  # help function
    # Note that input is a DataFrame
    dims = df.shape[1]

    if dims == 1:
        # only the measure dim
        input_sum = sum(project_data(df, 0))
        pre_num.append(input_sum)

        df_out.loc[len(df_out)] = pre_num

    else:
        # the general case

        dim0_vals = set(project_data(df, 0).values)
        temp_pre_num = deepcopy(pre_num)
        for dim0_v in dim0_vals:
            pre_num = deepcopy(temp_pre_num)
            sub_data = slice_data_dim0(df, dim0_v)
            pre_num.append(dim0_v)

            _buc_rec_optimized(sub_data, pre_num, df_out)
        ## for R_{ALL}
        sub_data = remove_first_dim(df)

        pre_num = deepcopy(temp_pre_num)
        pre_num.append("ALL")
        _buc_rec_optimized(sub_data, pre_num, df_out)
        
def buc_rec_optimized(df):  # do not change the heading of the function

    if df.shape[0] == 1:
        df_out = one_dim_duc(df)
    else:
        header = list(df)
        df_out = pd.DataFrame(columns=header)
        _buc_rec_optimized(df, [], df_out)
    return df_out

################# Question 2 #################

def v_opt_dp(x, num_bins):# do not change the heading of the function
    pass # **replace** this line with your code