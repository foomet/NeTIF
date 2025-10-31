import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import argparse
from sklearn.feature_selection import mutual_info_classif
from FCBF_module import FCBF, FCBFK, FCBFiP, get_i
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import QuantileTransformer
import numpy as np
import pandas as pd
import os
import cv2
import math
import random
import matplotlib.pyplot as plt
import shutil
from sklearn.preprocessing import QuantileTransformer
from PIL import Image
import warnings
import time
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import operator
import numpy as np
from PIL import Image
from collections import defaultdict
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.metrics import accuracy_score
import os

# Python Version 3.6.15

def put_tabs(tabsize):
    tabstring = ""

    for i in range(tabsize):
        tabstring += "\t"

    return tabstring
    
def print_tab(text, tabsize):
    print(put_tabs(tabsize) + text)


def get_label_map():
    
    labels = {'BENIGN', 'DDoS', 'DoS Hulk', 'FTP-Patator', 'SSH-Patator', 'Bot', 'DoS Slowhttptest', 'DoS slowloris', 'PortScan'}
    label_map = {0: 'BENIGN', 1: 'Bot', 2: 'DDoS', 3: 'DoS Hulk', 4: 'DoS Slowhttptest', 
             5: 'DoS slowloris', 6: 'FTP-Patator', 7: 'PortScan', 8: 'SSH-Patator'}
        
    return labels, label_map


def get_relevant_features(df_path):

    df = pd.read_csv(df_path)
        
    labels, label_map = get_label_map()

    df = df[df['Label'].isin(labels)]

    # Removes   'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 
    #           'Protocol', Timestamp'
    df = df.iloc[:, 7:]

    start_time = time.time()
    print(f"Processing 'get_relevant_features'...")
    tab_level = 1

    # Z-score normalization
    # print_tab('Z-score normalizing', tab_level)
    print_tab('Z-score normalizing...', tab_level)
    df['Label'] = df['Label'].astype('object')

    features = df.dtypes[df.dtypes != 'object'].index
    df[features] = df[features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    
    # Fill empty values by 0
    df = df.fillna(0)

    print_tab('Encoding labels...', tab_level)
    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])

    
    df_major = df[(df['Label'].isin([0,3,2]))]
    df_minor = df.drop(df_major.index)
    

    X = df_major.drop(['Label'],axis=1) 
    y = df_major.iloc[:, -1].values.reshape(-1,1)
    y=np.ravel(y)
    
    # use k-means to cluster the data samples and select a proportion of data from each cluster
    print_tab('Clustering data for undersampling...', tab_level)
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=1000, random_state=0).fit(X)

    klabel=kmeans.labels_
    df_major['klabel']=klabel

    cols = list(df_major)
    cols.insert(78, cols.pop(cols.index('Label')))
    df_major = df_major.loc[:, cols]

    def typicalSampling(group):
        name = group.name
        # frac = 0.008
        frac = 0.1
        return group.sample(frac=frac, random_state=0)

    print_tab('Underesampling majority classes...', tab_level)
    result = df_major.groupby(
        'klabel', group_keys=False
    ).apply(typicalSampling)

    result = result.drop(['klabel'],axis=1)
    result = result.append(df_minor)
    
    result.to_csv(f'./CICIDS_flow_undersampled.csv',index=False)

    df = result

    df['Label'] = df['Label'].astype('object')
    features = df.dtypes[df.dtypes != 'object'].index
    df['Label'] = df['Label'].astype('int')

    X = df.drop(['Label'],axis=1).values
    y = df.iloc[:, -1].values.reshape(-1,1)
    y=np.ravel(y)

    print_tab('Splitting training and testing data...', tab_level)
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0)

    importances = mutual_info_classif(X_train, y_train)

    # calculate the sum of importance scores
    f_list = sorted(zip(map(lambda x: round(x, 4), importances), features), reverse=True)
    Sum = 0
    fs = []
    for i in range(0, len(f_list)):
        Sum = Sum + f_list[i][0]
        fs.append(f_list[i][1])

    # select the important features from top to bottom until the accumulated importance reaches 90%
    f_list2 = sorted(zip(map(lambda x: round(x, 4), importances/Sum), features), reverse=True)
    Sum2 = 0
    fs = []
    for i in range(0, len(f_list2)):
        Sum2 = Sum2 + f_list2[i][0]
        fs.append(f_list2[i][1])
        if Sum2>=0.9:
            break        

    X_fs = df[fs].values

    fcbf = FCBFK(k = 20)


    print_tab('Selecting top 20 features...', tab_level)
    X_fss = fcbf.fit_transform(X_fs,y)

    selected_features = fcbf.idx_sel

    print_tab(f'Selected features: {selected_features}', tab_level)

    end_time = time.time()
    print_tab(f'Time taken: {end_time-start_time} seconds', tab_level)
    
    return selected_features
    

def transform_data(selected_features):

    print(f"Processing 'transform_data'...")

    start_time = time.time()
    tab_level = 1
    df = pd.read_csv(f'CICIDS_flow_undersampled.csv')

    # Append the "label" column.
    selected_features.append(df.shape[1] - 1)

    df = df.iloc[:, selected_features]

    # Transform all features into the scale of [0,1]
    print_tab("Quantile transformation...", tab_level)

    df['Label'] = df['Label'].astype('object')

    numeric_features = df.dtypes[df.dtypes != 'object'].index
    scaler = QuantileTransformer() 
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    df['Label'] = df['Label'].astype('int')
    
    # Multiply the feature values by 255 to transform them into the scale of [0,255]
    print_tab("Transforming scale...", tab_level)
    df[numeric_features] = df[numeric_features].apply(
        lambda x: (x*255))
    

    df.to_csv(f'CICIDS_flows_processed.csv', index=False)

    end_time = time.time()
    print_tab(f'Time taken: {end_time-start_time} seconds', tab_level)


if __name__ == "__main__":
    
    total_start_time = time.time()

    start_df_path = '/home/kdd1/shared_folder/CICIDS2017/CICIDS2017_preprocessed.csv'

    # Requires Python 3.6 (version 3.6.15 is used)
    selected_features = get_relevant_features(df_path=start_df_path)
    transform_data(selected_features)

    
    total_end_time = time.time()

    print(f'Total elapsed time: {total_end_time-total_start_time} seconds')
