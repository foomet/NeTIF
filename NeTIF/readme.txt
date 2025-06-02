In order to replicate our NeTIF experiments, perform the following steps:

1.) Run "Preprocessing_of_X_Dataset.iypnb". This will provide the preprocessed dataset, required for the next step.
2.) Run "Pipeline.ipynb". This will generate the labelled PCAP files required for the next step.
3.) Run "X_preprocess.ipynb". This will apply the NeTIF data transformation.
4.) Run "NeTIF_X.py". This will train a CNN model on the data generated in step 3.
