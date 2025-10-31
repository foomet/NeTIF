import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os
import time
import numpy as np
import torch
import socket
from sklearn.preprocessing import LabelEncoder
import struct
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import glob
import matplotlib.pyplot as plt
from pathlib import Path
from Functions.Pipeline import pipeline
from Functions.Optimized_Parser_Labelling import pcap_parser
import hashlib as hl
import csv
from os import close
import logging
import argparse
import sys
from datetime import timezone
import math
import pickle
import random
from Functions.Optimized_Parser_Labelling import *
log = logging.getLogger()
log.setLevel(logging.DEBUG)
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler 
from Functions.Pipeline import *
from sklearn.metrics import precision_score, recall_score, f1_score




# Preprocessing (Step 1) ---------------------

def preprocessing_part1():

    print("Preprocessing step 1")

    output_file = "./CICIDS2017_preprocessed.csv"
    label_folder = "./TrafficLabelling"
    df1 = pd.read_csv(label_folder + "/Monday-WorkingHours.pcap_ISCX.csv")
    df2 = pd.read_csv(label_folder + "/Tuesday-WorkingHours.pcap_ISCX.csv")
    df3 = pd.read_csv(label_folder + "/Wednesday-workingHours.pcap_ISCX.csv")
    df4 = pd.read_csv(label_folder + "/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
    df5 = pd.read_csv(
        label_folder + "/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        encoding="cp1252",
        low_memory=False,
    )
    df6 = pd.read_csv(label_folder + "/Friday-WorkingHours-Morning.pcap_ISCX.csv")
    df7 = pd.read_csv(label_folder + "/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    df8 = pd.read_csv(label_folder + "/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

    df_p = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)

    df_p.columns = df_p.columns.str.strip()

    df_p.drop(columns=["Fwd Header Length.1"], inplace=True)
    df_p = df_p.drop(df_p[pd.isnull(df_p["Flow ID"])].index)

    df_p.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_p.dropna(inplace=True)

    df_p.drop_duplicates(inplace=True)

    df_p.to_csv(output_file, index=False)

# --------------------------------------------

# Preprocessing (Step 2) ---------------------

def run_payloadbyte_pipeline():
    
    print("Preprocessing step 2")

    # in_dir = "./GeneratedLabelledFlows/TrafficLabelling/"
    in_dir = "./PCAPs/"
    out_dir = "./CICIDS_results/"
    Dataset_name = "CICIDS"
    processed_csv_file = "./CICIDS2017_preprocessed.csv"

    # This will take a long time. Be prepared to have this run for more than 24 hours.
    print("PayloadByte running...")
    df = pipeline(in_dir, out_dir, Dataset_name, processed_csv_file)

    os.makedirs('./CICIDS_results/Labelled_pcap_file/Combined/', exist_ok=True)
    os.makedirs('./CICIDS_results/miniflow/', exist_ok=True)

# --------------------------------------------


# Preprocessing (Step 3) ---------------------

def preprocessing_part2(pcap_labelling = 1, param_p = 1):

    if pcap_labelling == 1:

        df_p=pd.read_csv('./CICIDS2017_preprocessed.csv')
        df_p=df_p[['Timestamp','Source IP','Destination IP','Destination Port','Source Port','Protocol','Label']]
        df_p.rename(columns={'Timestamp': 'stime', 'Source IP': 'srcip', 'Destination IP': 'dstip', 'Destination Port': 'dsport', 'Source Port': 'sport', 'Label': 'label','Protocol': 'protocol_m'}, inplace=True)

        #Converting time into epoch format along with the mitigation of AM/PM issue.
        df_p['stime']=df_p['stime'].apply(lambda x: (datetime.strptime(x, '%d/%m/%Y %H:%M')) if x.count(':') == 1 else  (datetime.strptime(x, '%d/%m/%Y %H:%M:%S')))
        df_p['stime']=df_p['stime'].apply(lambda x: int((datetime(x.year,x.month,x.day,(x.hour+15),x.minute,x.second,tzinfo=timezone.utc)).timestamp()) if (x.hour>=1)&(x.hour<=7) else int((datetime(x.year,x.month,x.day,(x.hour+3),x.minute,x.second,tzinfo=timezone.utc)).timestamp()))
        df_p=df_p.sort_values(by='stime')

        df_p['protocol_m'] = df_p['protocol_m'].astype(str)
        df_p['protocol_m']=df_p['protocol_m'].apply(lambda x: x.replace('6.0', 'tcp'))
        df_p['protocol_m']=df_p['protocol_m'].apply(lambda x: x.replace('17.0', 'udp'))
        df_p['protocol_m']=df_p['protocol_m'].apply(lambda x: x.replace('0.0', 'other'))

        def time_processing(combine, num):
            ##  DDoS LOIT (15:56 – 16:16)
            # combine.drop(combine[(combine.stime>=1499453760 )&(combine.stime<=1499454960 )&(combine.label=='BENIGN')].index,inplace=True)
            
            if num == 5:
                combine.drop(
                    combine[(combine['stime'] >= 1499453760) & (combine['stime'] <= 1499454960) & (combine['label'] == 'BENIGN')].index,
                    inplace=True)
                ## Port Scan (13:55-15:30)
                combine.drop(combine[(combine.stime >= 1499446500) & (combine.stime <= 1499452200) & (combine.label == 'BENIGN')].index,
                            inplace=True)
                ## Botnet ARES (10:02 a.m. – 11:02 a.m.)
                combine.drop(combine[(combine.stime >= 1499432520) & (combine.stime <= 1499436120) & (combine.label == 'BENIGN')].index,
                            inplace=True)
            elif num == 4:
                ## Thursday
                ## Infiltration – Dropbox download (15:04 – 15:45 p.m.)
                combine.drop(combine[(combine.stime >= 1499364240) & (combine.stime <= 1499366700) & (combine.label == 'BENIGN')].index,
                            inplace=True)
                ## Infiltration – Dropbox download and (14:33 -14:35)
                combine.drop(combine[(combine.stime >= 1499362380) & (combine.stime <= 1499362500) & (combine.label == 'BENIGN')].index,
                            inplace=True)
                ## Web Attack – Sql Injection (10:40 – 10:42 a.m.)
                combine.drop(combine[(combine.stime >= 1499348400) & (combine.stime <= 1499348520) & (combine.label == 'BENIGN')].index,
                            inplace=True)
                ## Web Attack – XSS (10:15 – 10:35 a.m.)
                combine.drop(combine[(combine.stime >= 1499346900) & (combine.stime <= 1499348100) & (combine.label == 'BENIGN')].index,
                            inplace=True)
                ## Web Attack – Brute Force (9:20 – 10 a.m.)
                combine.drop(combine[(combine.stime >= 1499343600) & (combine.stime <= 1499346000) & (combine.label == 'BENIGN')].index,
                            inplace=True)
            
            elif num == 3:
                ## Wednesday
                ## Heartbleed Port 444 (15:12 - 15:32)
                combine.drop(combine[(combine.stime >= 1499278320) & (combine.stime <= 1499279520) & (combine.label == 'BENIGN')].index,
                            inplace=True)
                ## DoS GoldenEye (11:10 – 11:23 a.m.)
                combine.drop(combine[(combine.stime >= 1499263800) & (combine.stime <= 1499264580) & (combine.label == 'BENIGN')].index,
                            inplace=True)
                ## DoS Hulk (10:43 – 11 a.m.)
                combine.drop(combine[(combine.stime >= 1499262180) & (combine.stime <= 1499263200) & (combine.label == 'BENIGN')].index,
                            inplace=True)
                ##DoS Slowhttptest (10:14 – 10:35 a.m.)
                combine.drop(combine[(combine.stime >= 1499260440) & (combine.stime <= 1499261700) & (combine.label == 'BENIGN')].index,
                            inplace=True)
                ## DoS slowloris (9:47 – 10:10 a.m.)
                combine.drop(combine[(combine.stime >= 1499258820) & (combine.stime <= 1499260200) & (combine.label == 'BENIGN')].index,
                            inplace=True)
            
            elif num == 2:
            
                ## Tuesday
                ## FTP-Patator (9:20 – 10:20 a.m.) 
                combine.drop(combine[(combine.stime >= 1499170800) & (combine.stime <= 1499174400) & (combine.label == 'BENIGN')].index,
                            inplace=True)
                ##SSH-Patator (14:00 – 15:00 p.m.)
                combine.drop(combine[(combine.stime >= 1499187600) & (combine.stime <= 1499191200) & (combine.label == 'BENIGN')].index,
                            inplace=True)


        # Monday

        df = pd.read_csv('./CICIDS_results\pcap_file_csv_parser\pcap_csv_1.csv', index_col=0)
        df = df.sort_values(by='stime')

        stime = df.stime[0]
        ltime = int(df.stime.tail(1))
        print("Start Value:", df.stime[0])
        print("End Value:", int(df.stime.tail(1)))

        df.drop(columns=['frame_num','stime','ltime','protocol_s'],inplace=True)

        aa = df_p[(df_p['stime'] >= stime) & (df_p['stime'] <= ltime)]

        df.protocol_m.value_counts()
        aa.protocol_m.value_counts()
        combine = df.merge(aa, left_on=['srcip', 'dstip', 'dsport', 'sport', 'protocol_m'],
                        right_on=['srcip', 'dstip', 'dsport', 'sport', 'protocol_m'])

        combine.drop_duplicates(inplace=True)
        time_processing(combine, 1)

        combine.to_csv('./CICIDS_results\Labelled_pcap_file\labelled_pcap_csv_1.csv', index=False)

        # Tuesday

        df = pd.read_csv('./CICIDS_results\pcap_file_csv_parser\pcap_csv_2.csv', index_col=0)
        df = df.sort_values(by='stime')
        # df=df.sort_values(by='stime')
        stime = df.stime[0]
        ltime = int(df.stime.tail(1))
        print("Start Value:", df.stime[0])
        print("End Value:", int(df.stime.tail(1)))

        df.drop(columns=['frame_num','stime','ltime','protocol_s'],inplace=True)

        aa = df_p[(df_p['stime'] >= stime) & (df_p['stime'] <= ltime)]

        df.protocol_m.value_counts()
        aa.protocol_m.value_counts()
        combine = df.merge(aa, left_on=['srcip', 'dstip', 'dsport', 'sport', 'protocol_m'],
                        right_on=['srcip', 'dstip', 'dsport', 'sport', 'protocol_m'])

        combine.drop_duplicates(inplace=True)
        time_processing(combine, 2)

        combine.to_csv('./CICIDS_results\Labelled_pcap_file\labelled_pcap_csv_2.csv', index=False)

        # Wednesday

        df = pd.read_csv('./CICIDS_results\pcap_file_csv_parser\pcap_csv_3.csv', index_col=0)
        df = df.sort_values(by='stime')

        stime = df.stime[0]
        ltime = int(df.stime.tail(1))
        print("Start Value:", df.stime[0])
        print("End Value:", int(df.stime.tail(1)))

        df.drop(columns=['frame_num','stime','ltime','protocol_s'],inplace=True)

        aa = df_p[(df_p['stime'] >= stime) & (df_p['stime'] <= ltime)]

        df.protocol_m.value_counts()
        aa.protocol_m.value_counts()

        combine = df.merge(aa, left_on=['srcip', 'dstip', 'dsport', 'sport', 'protocol_m'],
                        right_on=['srcip', 'dstip', 'dsport', 'sport', 'protocol_m'])

        combine.drop_duplicates(inplace=True)
        time_processing(combine, 3)

        combine.to_csv('./CICIDS_results\Labelled_pcap_file\labelled_pcap_csv_3.csv', index=False)

        # Thursday

        df = pd.read_csv('./CICIDS_results\pcap_file_csv_parser\pcap_csv_4.csv', index_col=0)
        df = df.sort_values(by='stime')

        stime = df.stime[0]
        ltime = int(df.stime.tail(1))
        print("Start Value:", df.stime[0])
        print("End Value:", int(df.stime.tail(1)))

        df.drop(columns=['frame_num','stime','ltime','protocol_s'],inplace=True)

        aa = df_p[(df_p['stime'] >= stime) & (df_p['stime'] <= ltime)]

        df.protocol_m.value_counts()
        aa.protocol_m.value_counts()
        combine = df.merge(aa, left_on=['srcip', 'dstip', 'dsport', 'sport', 'protocol_m'],
                        right_on=['srcip', 'dstip', 'dsport', 'sport', 'protocol_m'])

        combine.drop_duplicates(inplace=True)
        time_processing(combine, 4)

        combine.to_csv('./CICIDS_results\Labelled_pcap_file\labelled_pcap_csv_4.csv', index=False)

        # Friday

        df = pd.read_csv('./CICIDS_results\pcap_file_csv_parser\pcap_csv_5.csv', index_col=0)
        df = df.sort_values(by='stime')

        stime = df.stime[0]
        ltime = int(df.stime.tail(1))
        print("Start Value:", df.stime[0])
        print("End Value:", int(df.stime.tail(1)))

        df.drop(columns=['frame_num','stime','ltime','protocol_s'],inplace=True)

        aa = df_p[(df_p['stime'] >= stime) & (df_p['stime'] <= ltime)]

        df.protocol_m.value_counts()
        aa.protocol_m.value_counts()
        combine = df.merge(aa, left_on=['srcip', 'dstip', 'dsport', 'sport', 'protocol_m'],
                        right_on=['srcip', 'dstip', 'dsport', 'sport', 'protocol_m'])

        combine.drop_duplicates(inplace=True)
        time_processing(combine, 5)

        combine.to_csv('./CICIDS_results\Labelled_pcap_file\labelled_pcap_csv_5.csv', index=False)

        def combine_CICIDS(in_file_path,out_path):
            combine=pd.DataFrame(columns=['srcip', 'sport', 'dstip', 'dsport', 'protocol_m',
            'sttl', 'total_len', 'payload', 't_delta', 'stime','label'])
            for files in in_file_path:
                df=pd.read_csv(files)
                combine=combine.append(df, ignore_index=True)
                print(combine.shape)
            csv_out=out_path+"combined_labelled_pcap_csv.csv"
            logging.info("Exporting_combined_csv_file....")
            combine.to_csv(csv_out,index=False)
            return combine
        
        
        label_csv= './CICIDS_results\Labelled_pcap_file\\'
        in_file=[]
        for x in range(1,6): ## Starting & Ending values for the files
            in_file.append(label_csv+"labelled_pcap_csv_"+str(x)+".csv")

        out_path= './CICIDS_results\Labelled_pcap_file\Combined\\'


        df_payload=combine_CICIDS(in_file,out_path)

        df_payload.drop_duplicates(inplace=True)
        df_payload.drop(df_payload[df_payload.payload.isnull()].index,inplace=True)

        x=df_payload['payload']
        new=[]
        for p in range(len(x)):
            try:
                o=(int((x.iloc[p]), 16))
                if o>0:
                    new.append(1)
                else:
                    new.append(0)
            except Exception as e:
                print(p)
                new.append('Big')


        df_payload['payload_int']=new

        df_payload.drop(df_payload[df_payload.payload_int==0].index,inplace=True)
        df_payload.pop('payload_int')

        df_payload.sttl=df_payload.sttl.astype('int32')
        df_payload.dsport=df_payload.dsport.astype('int32')
        df_payload.sport=df_payload.sport.astype('int32')
        df_payload.total_len=df_payload.total_len.astype('int32')

        dict1={ 'BENIGN': 362108,
        'DoS Hulk':          250000,
        'DDoS'  :         214200,
        'DoS GoldenEye':     128122,
        'DoS slowloris':    121097,
        'Infiltration'        :103700,
        'DoS Slowhttptest'         :  80542,
        'SSH-Patator':          48165,
        'FTP-Patator'            :   31843,
        'Heartbleed'              :  13486,
        'Web Attack - Brute Force'            :   11700,
        'Web Attack - XSS'              :  3300,
        'Bot'            :   2543,
        'PortScan'              :  830,
        'Web Attack - Sql Injection': 12
        }

        rus = RandomUnderSampler(random_state=42,sampling_strategy=dict1)

        X_res, y_res = rus.fit_resample(df_payload.iloc[:,:-1], df_payload.iloc[:,-1])

        X_res['label']=y_res
        y_res = X_res['label']

        le = LabelEncoder()
        label = le.fit_transform(X_res['protocol_m'])
        X_res['protocol']=label
        X_res['attack_cat']=y_res


        X_tr,Ytrain =payload_to_bytes(X_res,1500)

        X_tr = np.column_stack((X_tr,np.array(X_res.iloc[:,0])))
        X_tr = np.column_stack((X_tr,np.array(X_res.iloc[:,1])))
        X_tr = np.column_stack((X_tr,np.array(X_res.iloc[:,2])))
        X_tr = np.column_stack((X_tr,np.array(X_res.iloc[:,3])))
        X_tr = np.column_stack((X_tr,np.array(X_res.iloc[:,4])))
        X_tr = np.column_stack((X_tr,np.array(X_res.iloc[:,5])))
        X_tr = np.column_stack((X_tr,np.array(X_res.iloc[:,6])))
        X_tr = np.column_stack((X_tr,np.array(X_res.iloc[:,8])))
        X_tr = np.column_stack((X_tr,np.array(X_res.iloc[:,9])))


        name=[]
        for x in range(1,1501):
            name.append("payload_byte_"+str(x))
        name.append("srcip")
        name.append("srcport")
        name.append("dstip")
        name.append("dstport")
        name.append("protocol_m")
        name.append("ttl")
        name.append("total_len")
        name.append("t_delta")
        name.append("Timestamp")

        df = pd.DataFrame(X_tr, columns=name)
        df['label']=Ytrain

        # Miniflow Generation

        start_time = time.time()

        grouped_df = df.groupby(['srcip','srcport','dstip','dstport','protocol_m'], as_index=False)
        i=0
        for key, item in grouped_df:
            item.to_csv(f'./CICIDS_results/miniflow\miniflow-{i}.csv', index=False)
            i +=1

        end_time = time.time()
        print(f"Elapsed miniflow creation time: {end_time - start_time} seconds.")


    start_algo1_time = time.time()

    parameter_p = param_p

    os.makedirs(f'./CICIDS_results/packets-{parameter_p}/', exist_ok=True)

    folder_path = r'./CICIDS_results\miniflow' 
    output_path = f'./CICIDS_results\packets-{parameter_p}' 
    i = 1

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            if len(df) >= parameter_p:
                output_filename = 'packet-{}.csv'.format(i)
                output_file_path = os.path.join(output_path, output_filename)
                df = df.sort_values(by=['Timestamp'])
                selected_df = df.head(parameter_p)
                selected_df.to_csv(output_file_path, index=False)
                i += 1


    dfs = []
    for file in glob.glob(f'./CICIDS_results/packets-{parameter_p}/*.csv'):
        df = pd.read_csv(file)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    unique_protocols = all_df['protocol_m'].unique()
    protocol_map = {'tcp': 6, 'udp': 17, 'icmp': 1}
    all_df['protocol_m'] = all_df['protocol_m'].map(protocol_map)

    def create_flow_id(row):
        if row['label'] == 'BENIGN':
            return row['dstip'] + '-' + row['srcip'] + '-' + str(row['dstport']) + '-' + str(row['srcport']) + '-' + str(row['protocol_m'])
        else:
            return row['srcip'] + '-' + row['dstip'] + '-' + str(row['srcport']) + '-' + str(row['dstport']) + '-' + str(row['protocol_m'])
        
    all_df['flow_id'] = all_df.apply(create_flow_id, axis=1)

    # create a list of unique flow IDs
    flow_ids = all_df['flow_id'].unique()

    # create a dataframe with columns for flow number and flow ID
    df = pd.DataFrame({'flow_number': range(1, len(flow_ids)+1),
                    'flow_id': flow_ids})

    # save the dataframe to a CSV file
    df.to_csv(f'CICIDS-{parameter_p}-number-packets-flow_ids.csv', index=False)


    groups = all_df.groupby('flow_id')

    new_df = pd.DataFrame()

    # define a function to extract flow information
    def extract_flow_info(group):
        # take the mean of all three packet feature values for this flow
        flow_features = group.drop(['flow_id','srcip', 'srcport', 'dstip', 'dstport', 'Timestamp', 'label','protocol_m'], axis=1).mean(numeric_only=True)
        
        # extract the protocol of the flow
        flow_protocol = group['protocol_m'].unique()[0]
        
        # extract the label of the flow
        flow_label = group['label'].unique()[0]
        
        # return a new row with the mean features, the flow protocol, and the flow label
        return pd.concat([flow_features, pd.Series(flow_protocol), pd.Series(flow_label)])

    # group packets by their flow_id and apply the extract_flow_info function to each group
    new_df = all_df.groupby('flow_id').apply(extract_flow_info).reset_index(drop=True)
    print("Grouped packets by flow_id and extracted flow information...")

    # save the new dataframe to a new csv file

    new_df = new_df.rename(columns={'0': 'protocol_m','0.1': 'label'})

    new_df.to_csv(f'final_CICIDS_pequals{parameter_p}.csv', index=False)
    print("Saved the new dataframe to csv file...")

    end_algo1_time = time.time()
    print(f"Elapsed Algorithm 1 time: {end_algo1_time - start_algo1_time} seconds.")


# --------------------------------------------

# Training (Step 4) --------------------------

def training_and_testing(param_p = 1):

    df = pd.read_csv(f'./final_CICIDS_pequals{param_p}.csv')

    if df.columns[-1] != 'label':
        df = df.rename(columns={df.columns[-1]: 'label'})

    labels = {'BENIGN', 'DDoS', 'DoS Hulk', 'FTP-Patator', 'SSH-Patator', 'Bot', 'DoS Slowhttptest', 'DoS slowloris', 'PortScan'}
    label_map = {0: 'BENIGN', 1: 'Bot', 2: 'DDoS', 3: 'DoS Hulk', 4: 'DoS Slowhttptest', 
                5: 'DoS slowloris', 6: 'FTP-Patator', 7: 'PortScan', 8: 'SSH-Patator'}

    df = df[df['label'].isin(labels)]

    start_preprocess_time = time.time()

    string_cols = [col for col in df.columns if df[col].dtype == object]
    le = LabelEncoder()
    df[string_cols] = df[string_cols].apply(lambda col: le.fit_transform(col))

    # Separate the features and labels
    features = df.iloc[:, :-1].values 
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    labels = df.iloc[:, -1].values

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=4)

    # Print the size of each set
    print("Training features shape: ", train_features.shape)
    print("Training labels shape: ", train_labels.shape)
    print("Testing features shape: ", test_features.shape)
    print("Testing labels shape: ", test_labels.shape)


    # Count the number of instances for each label in the training set
    unique_labels_train, counts_train = np.unique(train_labels, return_counts=True)
    for label, count in zip(unique_labels_train, counts_train):
        print(f"Training set: Label {label} has {count} instances")

    # Count the number of instances for each label in the testing set
    unique_labels_test, counts_test = np.unique(test_labels, return_counts=True)
    for label, count in zip(unique_labels_test, counts_test):
        print(f"Testing set: Label {label} has {count} instances")


    # Convert the training and testing data into PyTorch Tensors
    train_features_tensor = torch.from_numpy(train_features).float()
    train_labels_tensor = torch.from_numpy(train_labels).long()
    test_features_tensor = torch.from_numpy(test_features).float()
    test_labels_tensor = torch.from_numpy(test_labels).long()

    # Create a PyTorch TensorDataset for the training data
    train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)

    test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)

    end_preprocess_time = time.time()

    total_preprocessing_time = end_preprocess_time - start_preprocess_time
    print(f"Elapsed preprocessing time: {total_preprocessing_time} seconds")

    # Create a PyTorch DataLoader for the training data
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=(1,3), padding=(0,1))
            self.conv2 = nn.Conv2d(32, 64, kernel_size=(1,3), padding=(0,1))
            self.fc1 = nn.Linear(64 * 1504, 128)
            # self.fc2 = nn.Linear(128, 11)
            self.fc2 = nn.Linear(128, args.numclass)

        def forward(self, x):
            x = x.view(-1, 1, 1504, 1)
            x = self.conv1(x)
            x = nn.functional.relu(x)
            x = self.conv2(x)
            x = nn.functional.relu(x)
            x = x.view(-1, 64 * 1504)
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.fc2(x)
            return x


    # Create an instance of the model and pass a batch of inputs
    model = Net()
    inputs, targets = next(iter(train_dataloader))

    outputs = model(inputs)
    # find the unique labels in the target tensor
    unique_labels = torch.unique(train_labels_tensor)

    # print the unique labels
    print(unique_labels)
    # Print the shape of the output tensor
    print(outputs.shape)


    start_training_testing_time = time.time()

    model = Net()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model for 10 epochs
    for epoch in range(30):
        for i, (inputs, labels) in enumerate(train_dataloader):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Print the loss every 100 batches
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, 30, i+1, len(train_dataloader), loss.item()))

    torch.save(model.state_dict(), f'cicids_{param_p}.pth')


    # validate the model
    correct = 0
    total = 0
    with torch.no_grad():
            for data in test_dataloader:
                images, labels = data
                images, labels = images, labels
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    print("Accuracy of the network on the test data: {:.2f}%".format(100 * correct / total))

    print("Finished training")

    
    # Make predictions on the test set
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in test_dataloader:
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(y.numpy())
            y_pred.extend(predicted.numpy())

    # Compute precision, recall, and F1 score
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

    print('Precision:', precision*100)
    print('Recall:', recall*100)
    print('F1 score:', f1*100)

    # Convert y_true and y_pred to lists
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)

    # Map the label values to their corresponding names using list comprehension
    y_true_mapped = [label_map[y] for y in y_true_list]
    y_pred_mapped = [label_map[y] for y in y_pred_list]

    # Generate the classification report
    print(classification_report(y_true_mapped, y_pred_mapped, zero_division=1))
    print('\n')
    print('                    Confusion Matrix')




    end_training_testing_time = time.time()

    total_training_testing_time = end_training_testing_time - start_training_testing_time

    print(f"Elapsed training and testing time: {total_training_testing_time:.2f} seconds.")



# --------------------------------------------


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--payloadbyte", type=int, default=1, help="Decides if the PayloadByte process and it's requisite preprocessing occurs. 0 for no, 1 for yes.")
    parser.add_argument("--label", type=int, default=1, help="Decides if the PCAP labelling process occurs. This only needs to be done for the first execution. 0 for no, 1 for yes.")
    parser.add_argument("--p", type=int, default=1, help="The value of parameter P.")
    parser.add_argument("--eval", type=int, default=0, help="Decides if -only- training and testing occurs, without any data processing.")
    args = parser.parse_args()

    if args.eval != 1:

        if args.payloadbyte == 1:
            preprocessing_part1()
            run_payloadbyte_pipeline()

        preprocessing_part2(pcap_labelling = args.label, param_p = args.p)
        training_and_testing(param_p = args.p)
    else:
        training_and_testing(param_p = args.p)

