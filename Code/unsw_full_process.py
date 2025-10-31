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
from datetime import datetime, timezone, timedelta
from imblearn.under_sampling import RandomUnderSampler 
import os
import pandas as pd
import numpy as np
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler 
from sklearn.preprocessing import LabelEncoder
from Functions.Pipeline import *



# Preprocessing (Step 1) ---------------------

def preprocessing_part1():

    print("Preprocessing step 1")

    input_dir = "./UNSW_CSVs/"
    output_file = "./UNSW-NB15_preprocessed.csv"

    # Reading of all 4 csv files of UNSW
    dfs = []
    for i in range(1, 5):
        path = input_dir + f"/UNSW-NB15_{i}.csv"  # There are 4 input csv files
        dfs.append(pd.read_csv(path, header=None, low_memory=False))
    all_data = pd.concat(dfs).reset_index(drop=True)

    # Adding Column names to the CSV file
    df_col = pd.read_csv(input_dir + "/NUSW-NB15_features.csv", encoding="ISO-8859-1")
    df_col["Name"] = df_col["Name"].apply(lambda x: x.strip().replace(" ", "").lower())
    all_data.columns = df_col["Name"]


    all_data.drop(columns=["ct_flw_http_mthd", "is_ftp_login"], inplace=True)

    # Replacing Missing value with normal in attack_cat column. Moreover, removing adddtional spaces and converting labels to lower case
    all_data["attack_cat"] = all_data.attack_cat.fillna(value="normal").apply(lambda x: x.strip().lower())

    all_data["attack_cat"] = all_data["attack_cat"].replace("backdoors", "backdoor", regex=True).apply(lambda x: x.strip().lower())

    # removing all the "-" and replacing those with "None"
    all_data["service"] = all_data["service"].apply(lambda x: "None" if x == "-" else x)

    # One or more values are in string, converting to int
    all_data["ct_ftp_cmd"] = all_data["ct_ftp_cmd"].apply(lambda x: 0 if x == " " else x).astype(int)

    all_data = all_data.drop_duplicates(inplace=True)

    Important_protocol = all_data.proto[all_data.label == 1].value_counts()

    b = Important_protocol.keys()[25:]
    ## Retain these protocols
    b = b.drop("gmtp")
    b = b.drop("ipip")
    b = b.drop("larp")
    b = b.drop("dgp")
    b = b.drop("pnni")
    b = b.drop("fc")
    b = b.drop("iplt")
    b = b.drop("pipe")
    b = b.drop("sps")
    b = b.drop("sccopmce")
    b = b.drop("crudp")
    b = b.drop("crtp")
    b = b.drop("fire")
    b = b.drop("rvd")
    b = b.drop("rdp")
    b = b.drop("hmp")
    b = b.drop("pup")
    b = b.drop("egp")
    b = b.drop("ip")
    b = b.drop("ib")

    all_data["proto"].replace(b, "others", inplace=True)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    all_data.to_csv(output_file, index=False)




# --------------------------------------------

# Preprocessing (Step 2) ---------------------

def run_payloadbyte_pipeline():
    
    print("Preprocessing step 2")

    
    in_dir = "./UNSW_pcap_files/"
    out_dir = "./UNSW_results/"
    Dataset_name = "UNSW"
    processed_csv_file = "./UNSW-NB15_preprocessed.csv"

    # This will take a long time. Be prepared to have this run for more than 24 hours.
    print("PayloadByte running...")
    df = pipeline(in_dir, out_dir, Dataset_name, processed_csv_file)

    os.makedirs('./UNSW_results/Labelled_pcap_file/Combined/', exist_ok=True)
    os.makedirs('./UNSW_results/miniflow/', exist_ok=True)

# --------------------------------------------


# Preprocessing (Step 3) ---------------------

def preprocessing_part2(pcap_labelling = 1, param_p = 1):

    if pcap_labelling == 1:

        df_p=pd.read_csv('./UNSW-NB15_preprocessed.csv')
        df_p.rename(columns={'proto':'protocol_m'}, inplace=True)

        # Convert UNIX timestamp to datetime, adjust time, and convert back to UNIX
        def adjust_time(unix_time):
            dt = datetime.fromtimestamp(unix_time, tz=timezone.utc)
            if 1 <= dt.hour <= 7:
                dt += timedelta(hours=15)
            else:
                dt += timedelta(hours=3)
            return int(dt.timestamp())

        df_p['stime'] = df_p['stime'].apply(adjust_time)

        df_p=df_p.sort_values(by='stime')

        def combine_UNSW(in_folder_path, out_path):
            # Initialize empty DataFrame with expected columns
            combine = pd.DataFrame(columns=[
                'srcip', 'sport', 'dstip', 'dsport', 'protocol_m',
                'sttl', 'total_len', 'payload', 't_delta', 'stime', 'label'
            ])

            # Discover all labelled CSV files
            for root, dirs, files in os.walk(in_folder_path):
                for file in files:
                    if file.startswith('labelled_') and file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        df = pd.read_csv(file_path)
                        combine=combine.append(df, ignore_index=True)
                        print(combine.shape)

            # Save combined result
            os.makedirs(out_path, exist_ok=True)
            csv_out = os.path.join(out_path, "combined_labelled_pcap_csv.csv")
            logging.info("Exporting combined CSV file....")
            combine.to_csv(csv_out, index=False)
            return combine
        
        label_csv_folder = './UNSW_results/Labelled_pcap_file/'
        out_combined_path = './UNSW_results/Labelled_pcap_file/Combined/'

        df_payload = combine_UNSW(label_csv_folder, out_combined_path)

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

        df_payload.to_csv(r'./UNSW_results\Labelled_pcap_file\Combined\preprocessed_pcap_data.csv',index=False)

        df_payload= pd.read_csv(r'./UNSW_results\Labelled_pcap_file\Combined\preprocessed_pcap_data.csv')
        
        df_payload.drop(columns=['label'], inplace=True)
        df_payload.rename(columns={'attack_cat': 'label'}, inplace=True)

        dict1={ 'normal': 50000
        }

        rus = RandomUnderSampler(random_state=42,sampling_strategy=dict1)

        X_res, y_res = rus.fit_resample(df_payload.iloc[:,:-1], df_payload.iloc[:,-1])

        X_res['label']=y_res

        le = LabelEncoder()
        label = le.fit_transform(X_res['protocol_m'])
        X_res['protocol']=label

        X_res.drop("protocol_m", axis=1, inplace=True)
        X_res.drop("protocol_s", axis=1, inplace=True)
        X_res.drop("frame_num", axis=1, inplace=True)

        # X_res['attack_cat']=y_res
        X_res['label']=y_res

        X_tr,Ytrain =payload_to_bytes(X_res,1500)

        X_tr = np.column_stack((X_tr,np.array(X_res.iloc[:,0])))    # srcip
        print('Done with srcip')
        X_tr = np.column_stack((X_tr,np.array(X_res.iloc[:,1])))    # sport
        print('Done with sport')
        X_tr = np.column_stack((X_tr,np.array(X_res.iloc[:,2])))    # dstip
        print('Done with dstip')
        X_tr = np.column_stack((X_tr,np.array(X_res.iloc[:,3])))    # dsport
        print('Done with dsport')
        X_tr = np.column_stack((X_tr,np.array(X_res.iloc[:,4])))    # sttl
        print('Done with sttl')
        X_tr = np.column_stack((X_tr,np.array(X_res.iloc[:,5])))    # total_len 
        print('Done with total_len')
        # SKIP PAYLOAD
        X_tr = np.column_stack((X_tr,np.array(X_res.iloc[:,7])))    # t_delta
        print('Done with t_delta')
        X_tr = np.column_stack((X_tr,np.array(X_res.iloc[:,8])))    # stime
        print('Done with stime')
        X_tr = np.column_stack((X_tr,np.array(X_res.iloc[:,10])))   # SKIP ltime; go to proto
        print('Done with proto')

        name=[]
        for x in range(1,1501):
            name.append("payload_byte_"+str(x))
        name.append("srcip")
        name.append("srcport")
        name.append("dstip")
        name.append("dstport")
        name.append("ttl")
        name.append("total_len")
        name.append("t_delta")
        name.append("Timestamp")
        name.append("protocol_m")

        df = pd.DataFrame(X_tr, columns=name)
        df['label']=Ytrain

        # Miniflow Files

        grouped_df = df.groupby(['srcip','srcport','dstip','dstport','protocol_m'], as_index=False)
        i=0

        os.makedirs(f'./UNSW_results\miniflow/', exist_ok=True)

        start_time = time.time()

        for key, item in grouped_df:
            item.to_csv(f'./UNSW_results/miniflow/miniflow-{i}.csv', index=False)
            i +=1

        end_time = time.time()
        print(f"Elapsed miniflow creation time: {end_time - start_time} seconds.")


    start_algo1_time = time.time()

    parameter_p = param_p

    os.makedirs(f'./UNSW_results/packets-{parameter_p}/', exist_ok=True)

    folder_path = r'./UNSW_results\miniflow' # Replace with the path to your folder
    output_path = f'./UNSW_results\packets-{parameter_p}' # Replace with the path to your output folder
    i = 1

    os.makedirs(f'./UNSW_results\packets-{parameter_p}/', exist_ok=True)

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
    for file in glob.glob(f'./UNSW_results/packets-{parameter_p}/*.csv'):
        df = pd.read_csv(file)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    unique_protocols = all_df['protocol_m'].unique()

    protocol_map = {
    'icmp': 1,
    'igmp': 2,
    'ipip': 4,       # IP in IP encapsulation
    'tcp': 6,
    'udp': 17,
    'gre': 47,
    'esp': 50,
    'ah': 51,
    'egp': 8,
    'ospf': 89,
    'sctp': 132,
    'rsvp': 46,
    'pim': 103,
    'ax.25': 93,
    'ip': 0,
    'ipv6': 41,
    'vmtp': 81,
    'sun-nd': 77,
    'dgp': 86,
    'hmp': 20,
    'nvp': 11,
    'snp': 109,
    'sps': 83,
    'unas': 106,
    'rdp': 27,
    'crtp': 126,
    'crudp': 127,
    'sep': 33,
    'pipe': 131,
    'pnni': 102,
    'mobile': 55,
    'fire': 83,
    'micp': 95,
    'secure-vmtp': 92,
    'iplt': 129,
    'ib': 114,
    'etherip': 97,
    'others': -1,      # dataset-specific
    'emcon': 14,
    'pup': 12,
    'sccopmce': 128,     # not in IANA list , but sscopmce is -- assume typo.
    'arp': -1           # not in IANA list  - treat as 'other'
    }

    all_df['protocol_m'] = all_df['protocol_m'].map(protocol_map)

    def create_flow_id(row):
        if row['label'] == 'BENIGN':
            return row['dstip'] + '-' + row['srcip'] + '-' + str(row['dstport']) + '-' + str(row['srcport']) + '-' + str(row['protocol_m'])
        else:
            return row['srcip'] + '-' + row['dstip'] + '-' + str(row['srcport']) + '-' + str(row['dstport']) + '-' + str(row['protocol_m'])
            
    all_df['flow_id'] = all_df.apply(create_flow_id, axis=1)

    # create a list of unique flow IDs
    flow_ids = all_df['flow_id'].unique()


    # group packets by their flow_id
    groups = all_df.groupby('flow_id')

    # initialize an empty dataframe to store the extracted flow information
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
    # rename the protocol and label columns
    #new_df = new_df.rename(columns={0: 'protocol_m', 1: 'label'})

    # save the new dataframe to a new csv file
    # new_df.to_csv(f'UNSW-{parameter_p}-number-packets-to-flows.csv', index=False)
    # print("Saved the new dataframe to csv file...")

    # rename the column 'A' to 'X'
    new_df = new_df.rename(columns={'0': 'protocol_m','0.1': 'label'})

    # save the new dataframe to a new csv file
    new_df.to_csv(f'final_UNSW_pequals{parameter_p}.csv', index=False)

    end_algo1_time = time.time()
    print(f"Elapsed Algorithm 1 time: {end_algo1_time - start_algo1_time} seconds.")







# --------------------------------------------

# Training (Step 4) --------------------------

def training_and_testing(param_p = 1):

    df = pd.read_csv(f'./final_UNSW_pequals{param_p}.csv')

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

    import numpy as np

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
            self.fc2 = nn.Linear(128, 10)

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

    torch.save(model.state_dict(), f'unsw_{args.p}.pth')


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

    import torch
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score
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

    label_map = {0: 'analysis', 1: 'backdoor', 2: 'dos', 3: 'exploits', 4: 'fuzzers', 5: 'generic', 6: 'normal', 7:'reconnaissance', 8: 'shellcode', 9: 'worms'}



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

