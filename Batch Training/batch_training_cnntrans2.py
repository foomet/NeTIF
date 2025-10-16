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
from tensorflow.keras.layers import Dense,Flatten,GlobalAveragePooling2D,Input,Conv2D,MaxPooling2D,Dropout,UpSampling2D
from tensorflow.keras.applications.xception import  Xception
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
import warnings
warnings.filterwarnings("ignore")
import tensorflow.keras as keras
import cv2
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from tensorflow.keras.layers import concatenate,Dense,Flatten,Dropout,Average
import operator
import numpy as np
from PIL import Image
from collections import defaultdict
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model,load_model
from tensorflow.keras import Input
from tensorflow.keras.layers import concatenate,Dense,Flatten,Dropout
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
import tensorflow.keras.callbacks as kcallbacks
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.utils import resample


def put_tabs(tabsize):
    tabstring = ""

    for i in range(tabsize):
        tabstring += "\t"

    return tabstring
    
def print_tab(text, tabsize):
    print(put_tabs(tabsize) + text)


def get_label_map(numclass):
    if numclass == 9:
        labels = {'BENIGN', 'DDoS', 'DoS Hulk', 'FTP-Patator', 'SSH-Patator', 'Bot', 'DoS Slowhttptest', 'DoS slowloris', 'PortScan'}
        label_map = {0: 'BENIGN', 1: 'Bot', 2: 'DDoS', 3: 'DoS Hulk', 4: 'DoS Slowhttptest', 
                5: 'DoS slowloris', 6: 'FTP-Patator', 7: 'PortScan', 8: 'SSH-Patator'}

    elif numclass == 8:
        labels = {'BENIGN', 'DDoS', 'DoS Hulk', 'FTP-Patator', 'SSH-Patator', 'Bot', 'DoS Slowhttptest', 'DoS slowloris'}
        label_map = {0: 'BENIGN', 1: 'Bot', 2: 'DDoS', 3: 'DoS Hulk', 4: 'DoS Slowhttptest', 
                5: 'DoS slowloris', 6: 'FTP-Patator', 7: 'SSH-Patator'}
        
    return labels, label_map


def generate_images(numclass):
    tab_level = 1
    print("Processing 'generate_images'...")
    
    df = pd.read_csv(f'./{numclass}class_CICIDS_flows_processed.csv')
    

    dfs = []

    target_size = 7000  # a compromise between smallest and largest

    for cls in df['Label'].unique():
        class_df = df[df['Label'] == cls]
        if len(class_df) > target_size:
            class_df = resample(class_df, replace=False, n_samples=target_size, random_state=42)
        else:
            class_df = resample(class_df, replace=True, n_samples=target_size, random_state=42)
        dfs.append(class_df)

    balanced_df = pd.concat(dfs)

    # Randomly shuffle the dataframe
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)    

    print_tab("Generating images...", tab_level)
    count=0
    ims = []

    image_count = 0

    label_count = {i: 0 for i in range(0,numclass)}

    for i in range(0, len(balanced_df)):  
        count=count+1
        if count<=60: 
            im=balanced_df.iloc[i].drop('Label').values
            label_count[balanced_df.iloc[i]['Label']] += 1
            ims=np.append(ims,im)
        else:
            
            max_value = max(label_count.values())  # maximum value
            max_keys = [k for k, v in label_count.items() if v == max_value] # getting all keys containing the `maximum`
            
            this_label = max_keys[0]
            image_path = f"{numclass}class_test_shuffled/{this_label}/"
            
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            
            image_count = image_count + 1
            ims=np.array(ims).reshape(20,20,3)
            array = np.array(ims, dtype=np.uint8)
            new_image = Image.fromarray(array)
            new_image.save(image_path+str(i)+'.png')
            count=0
            ims = []
            label_count = {i: 0 for i in range(0,numclass)}
   
    

        #resize the images 224*224 for better CNN training
    def get_224(folder,dstdir):
        imgfilepaths=[]
        for root,dirs,imgs in os.walk(folder):
            for thisimg in imgs:
                thisimg_path=os.path.join(root,thisimg)
                imgfilepaths.append(thisimg_path)
        for thisimg_path in imgfilepaths:
            dir_name,filename=os.path.split(thisimg_path)
            dir_name=dir_name.replace(folder,dstdir)
            new_file_path=os.path.join(dir_name,filename)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            img=cv2.imread(thisimg_path)
            img=cv2.resize(img,(224,224))
            cv2.imwrite(new_file_path,img)
        print('Finish resizing'.format(folder=folder))

    DATA_DIR2_224=f'./{numclass}class_test_224_shuffled/'
    get_224(folder=f'./{numclass}class_test_shuffled/',dstdir=DATA_DIR2_224)
    

def training_and_testing(numclass, train_constituent_models=True):

    print("Processing 'training_and_testing'...")

    tab_level = 1

    TARGET_SIZE=(224,224)     
    INPUT_SIZE=(224,224,3)    
    BATCHSIZE=32	

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_directory(
            f'./{numclass}class_train_224/',
            target_size=TARGET_SIZE,
            batch_size=BATCHSIZE,
            class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
            f'./{numclass}class_test_224/',
            target_size=TARGET_SIZE,
            batch_size=BATCHSIZE,
            class_mode='categorical')

    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = {'batch':[], 'epoch':[]}
            self.accuracy = {'batch':[], 'epoch':[]}
            self.val_loss = {'batch':[], 'epoch':[]}
            self.val_acc = {'batch':[], 'epoch':[]}
        def on_batch_end(self, batch, logs={}):
            self.losses['batch'].append(logs.get('loss'))
            self.accuracy['batch'].append(logs.get('acc'))
            self.val_loss['batch'].append(logs.get('val_loss'))
            self.val_acc['batch'].append(logs.get('val_acc'))
        def on_epoch_end(self, batch, logs={}):
            self.losses['epoch'].append(logs.get('loss'))
            self.accuracy['epoch'].append(logs.get('acc'))
            self.val_loss['epoch'].append(logs.get('val_loss'))
            self.val_acc['epoch'].append(logs.get('val_acc'))
        def loss_plot(self, loss_type):
            iters = range(len(self.losses[loss_type]))
            plt.figure()
            plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
            if loss_type == 'epoch':
                # acc
                plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
                # loss
                plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
                # val_acc
                plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
                # val_loss
                plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
            plt.grid(True)
            plt.xlabel(loss_type)
            plt.ylabel('acc-loss')
            plt.legend(loc="upper right")
            plt.savefig(f'./figs/{self.savepath}')
            # plt.show()

    history_this = LossHistory()


    if train_constituent_models:
        history_this.savepath = "losshistory_xception.png"

        def xception( num_class, epochs,savepath=f'./{numclass}class_xception.keras',history=history_this,input_shape=INPUT_SIZE):
            model_fine_tune = Xception(include_top=False, weights='imagenet', input_shape=input_shape)
            for layer in model_fine_tune.layers[:121]:		
                layer.trainable = False
            for layer in model_fine_tune.layers[121:]:
                layer.trainable = True

            model = GlobalAveragePooling2D()(model_fine_tune.output)
            
            model=Dense(units=256,activation='relu')(model)
            model=Dropout(0.5)(model)
            model = Dense(num_class, activation='softmax')(model)
            model = Model(model_fine_tune.input, model, name='xception')
            opt = keras.optimizers.Adam(learning_rate=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            #train model
            earlyStopping = kcallbacks.EarlyStopping(
                monitor='val_accuracy', patience=3, verbose=1, mode='auto')	
            saveBestModel = kcallbacks.ModelCheckpoint(
                filepath=savepath,
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True,
                mode='auto')
            hist = model.fit(
                train_generator,
                steps_per_epoch=len(train_generator),
                epochs=epochs,
                # class_weight=class_weights,
                validation_data=validation_generator,
                validation_steps=len(validation_generator),
                #use_multiprocessing=True, 
                callbacks=[earlyStopping, saveBestModel, history],
            )

        print_tab("Training xception...", tab_level)
        xception(num_class=numclass,epochs=20) 

        
        history_this.savepath = "losshistory_vgg16.png"

        def vgg16( num_class, epochs,savepath=f'./{numclass}class_VGG16.keras',history=history_this,input_shape=INPUT_SIZE):
            
            model_fine_tune = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
            
            for layer in model_fine_tune.layers[:15]:
                layer.trainable = False
            for layer in model_fine_tune.layers[15:]:
                layer.trainable = True
                
            
            
            model = GlobalAveragePooling2D()(model_fine_tune.output)
            
            model=Dense(units=256,activation='relu')(model)
            model=Dropout(0.5)(model)
            model = Dense(num_class, activation='softmax')(model)
            model = Model(model_fine_tune.input, model, name='vgg')
            opt = keras.optimizers.Adam(learning_rate=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)	
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])	
            
            earlyStopping = kcallbacks.EarlyStopping(
                monitor='val_accuracy', patience=3, verbose=1, mode='auto')	
            saveBestModel = kcallbacks.ModelCheckpoint(
                filepath=savepath,
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True,
                mode='auto')
            hist = model.fit(
                train_generator,
                steps_per_epoch=len(train_generator),
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=len(validation_generator),
                #use_multiprocessing=True, 
                #workers=2,
                callbacks=[earlyStopping, saveBestModel, history],
            )

        
        print_tab("Training vgg16...", tab_level)
        vgg16(num_class=numclass,epochs=20)

        history_this.savepath = "losshistory_vgg19.png"

        def vgg19( num_class, epochs,savepath=f'./{numclass}class_VGG19.keras',history=history_this,input_shape=INPUT_SIZE):
            model_fine_tune = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
            for layer in model_fine_tune.layers[:19]:	
                layer.trainable = False
            for layer in model_fine_tune.layers[19:]:
                layer.trainable = True
                

            
            model = GlobalAveragePooling2D()(model_fine_tune.output)
            
            model=Dense(units=256,activation='relu')(model)
            model=Dropout(0.5)(model)
            model = Dense(num_class, activation='softmax')(model)
            model = Model(model_fine_tune.input, model, name='vgg')
            opt = keras.optimizers.Adam(learning_rate=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)	
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])	

            earlyStopping = kcallbacks.EarlyStopping(
                monitor='val_accuracy', patience=3, verbose=1, mode='auto')	
            saveBestModel = kcallbacks.ModelCheckpoint(
                filepath=savepath,
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True,
                mode='auto')
            hist = model.fit(
                train_generator,
                steps_per_epoch=len(train_generator),
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=len(validation_generator),
                #use_multiprocessing=True, 
                #workers=2,
                callbacks=[earlyStopping, saveBestModel, history],
            )

        
        print_tab("Training vgg19...", tab_level)
        vgg19(num_class=numclass,epochs=20)	


        history_this.savepath = "losshistory_inception.png"    

        def inception( num_class, epochs,savepath=f'./{numclass}class_inception.keras',history=history_this,input_shape=INPUT_SIZE):
            model_fine_tune = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)
            for layer in model_fine_tune.layers[:148]:	
                layer.trainable = False
            for layer in model_fine_tune.layers[148:]:	
                layer.trainable = True
                

            model = GlobalAveragePooling2D()(model_fine_tune.output)
            
            model=Dense(units=256,activation='relu')(model)
            model=Dropout(0.5)(model)
            model = Dense(num_class, activation='softmax')(model)
            model = Model(model_fine_tune.input, model, name='resnet')
            opt = keras.optimizers.Adam(learning_rate=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)	
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 

            earlyStopping = kcallbacks.EarlyStopping(
                monitor='val_accuracy', patience=3, verbose=1, mode='auto')	#set early stop patience to save training time
            saveBestModel = kcallbacks.ModelCheckpoint(
                filepath=savepath,
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True,
                mode='auto')
            hist = model.fit(
                train_generator,
                steps_per_epoch=len(train_generator),
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=len(validation_generator),
                #use_multiprocessing=True, 
                callbacks=[earlyStopping, saveBestModel, history],
            )

        print_tab("Training inception...", tab_level)
        inception(num_class=numclass,epochs=20)	

        history_this.savepath = "losshistory_inres.png"

        def inceptionresnet( num_class, epochs,savepath=f'./{numclass}class_inceptionresnet.keras',history=history_this,input_shape=INPUT_SIZE):
            model_fine_tune = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
            for layer in model_fine_tune.layers[:522]:	
                layer.trainable = False
            for layer in model_fine_tune.layers[522:]:	
                layer.trainable = True
                

            
            model = GlobalAveragePooling2D()(model_fine_tune.output)
            
            model=Dense(units=256,activation='relu')(model)
            model=Dropout(0.5)(model)
            model = Dense(num_class, activation='softmax')(model)
            model = Model(model_fine_tune.input, model, name='resnet')
            opt = keras.optimizers.Adam(learning_rate=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)	
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 

            earlyStopping = kcallbacks.EarlyStopping(
                monitor='val_accuracy', patience=3, verbose=1, mode='auto')	#set early stop patience to save training time
            saveBestModel = kcallbacks.ModelCheckpoint(
                filepath=savepath,
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True,
                mode='auto')
            hist = model.fit(
                train_generator,
                steps_per_epoch=len(train_generator),
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=len(validation_generator),
                #use_multiprocessing=True, 
                callbacks=[earlyStopping, saveBestModel, history],
            )

        print_tab("Training inceptionresnet...", tab_level)
        inceptionresnet(num_class=numclass,epochs=20)


    #generate images from train set and validation set
    test_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = test_datagen.flow_from_directory(
            f'./{numclass}class_test_224/',
            target_size=TARGET_SIZE,
            batch_size=BATCHSIZE,
            class_mode='categorical')

    label=validation_generator.class_indices
    label={v: k for k, v in label.items()}


    rootdir = f'./{numclass}class_test_224/'


    test_laels = []
    test_images=[]
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if not (file.endswith(".jpeg"))|(file.endswith(".jpg"))|(file.endswith(".png")):
                continue
            test_laels.append(subdir.split('/')[-1])
            test_images.append(os.path.join(subdir, file))

    print_tab("Loading models...", tab_level)
    
    xception_model=load_model(f'./{numclass}class_xception.keras')
    # last_xception_layer = xception_model.layers[-1].name

    vgg_model=load_model(f'./{numclass}class_VGG16.keras')
    # last_vgg_layer = vgg_model.layers[-1].name

    vgg19_model=load_model(f'./{numclass}class_VGG19.keras')
    # last_vgg19_layer = vgg19_model.layers[-1].name

    incep_model=load_model(f'./{numclass}class_inception.keras')
    # last_incep_layer = incep_model.layers[-1].name

    inres_model=load_model(f'./{numclass}class_inceptionresnet.keras')
    # last_inres_layer = inres_model.layers[-1].name


    print_tab("Forming ensemble model...", tab_level)

    # model1=Model(inputs=[xception_model.layers[0].get_input_at(0)],outputs=xception_model.get_layer(last_xception_layer).output,name='xception')
    # model2=Model(inputs=[vgg_model.layers[0].get_input_at(0)],outputs=vgg_model.get_layer(last_vgg_layer).output,name='vgg')
    # model3=Model(inputs=[vgg19_model.layers[0].get_input_at(0)],outputs=vgg19_model.get_layer(last_vgg19_layer).output,name='vgg19')
    # model4=Model(inputs=[incep_model.layers[0].get_input_at(0)],outputs=incep_model.get_layer(last_incep_layer).output,name='incep')
    # model5=Model(inputs=[inres_model.layers[0].get_input_at(0)],outputs=inres_model.get_layer(last_inres_layer).output,name='inres')

    model1 = keras.Model(inputs=xception_model.input, outputs=xception_model.output, name='xception')
    model2 = keras.Model(inputs=vgg_model.input,      outputs=vgg_model.output,      name='vgg')
    model3 = keras.Model(inputs=vgg19_model.input,    outputs=vgg19_model.output,    name='vgg19')
    model4 = keras.Model(inputs=incep_model.input,    outputs=incep_model.output,    name='incep')
    model5 = keras.Model(inputs=inres_model.input,    outputs=inres_model.output,    name='inres')

    def lr_decay(epoch):
        lrs = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0001,0.00001,0.000001,
            0.000001,0.000001,0.000001,0.000001,0.0000001,0.0000001,0.0000001,0.0000001,0.0000001,0.0000001
            ]
        return lrs[epoch]
    
    auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    my_lr = LearningRateScheduler(lr_decay)


    ensemble_history = LossHistory()
    ensemble_history.savepath = "losshistory_ensemble.png"

    def ensemble(num_class,epochs,savepath=f'./{numclass}class_ensemble.keras'):
        img=Input(shape=(224,224,3),name='img')
        feature1=model1(img)
        feature2=model2(img)
        feature3=model3(img)
        feature4=model4(img)
        feature5=model5(img)
        x=concatenate([feature1,feature2,feature3,feature4,feature5])
        x=Dropout(0.5)(x)
        x=Dense(64,activation='relu')(x)
        x=Dropout(0.25)(x)
        output=Dense(num_class,activation='softmax',name='output')(x)
        model=Model(inputs=img,outputs=output)
        opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
        #train model
        earlyStopping=kcallbacks.EarlyStopping(monitor='val_accuracy',patience=2, verbose=1, mode='auto')
        saveBestModel = kcallbacks.ModelCheckpoint(filepath=savepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
        hist=model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=[earlyStopping,saveBestModel,ensemble_history,auto_lr],
        )


    print_tab("Training ensemble...", tab_level)
    ensemble_model=ensemble(num_class=numclass,epochs=20)

    ensemble_model=load_model(f'./{numclass}class_ensemble.keras')

    print_tab("Testing ensemble model...", tab_level)
    tab_level += 1

    #read images from validation folder

    
    rootdir = f'./{numclass}class_test_224/'

    test_laels = []
    test_images=[]
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if not (file.endswith(".jpeg"))|(file.endswith(".jpg"))|(file.endswith(".png")):
                continue
            test_laels.append(subdir.split('/')[-1])
            test_images.append(os.path.join(subdir, file))
            

    #test the averaging model on the validation set

    predict=[]
    length=len(test_images)
    t1 = time.time()
    for i in range((length//127)+1):
        inputimg=test_images[127*i:127*(i+1)]
        test_batch=[]
        for path in inputimg:
            thisimg=np.array(Image.open(path))/255
            test_batch.append(thisimg)
        #print(i, np.array(test_batch).shape)
        ensemble_model_batch=ensemble_model.predict(np.array(test_batch))
        predict_batch=list(np.argmax(ensemble_model_batch,axis=1))
        predict_batch=[label[con] for con in predict_batch]
        predict.append(predict_batch)

    predict=sum(predict,[])

    t2 = time.time()
    print_tab('The testing time is :%f seconds' % (t2-t1), tab_level)

    acc=accuracy_score(test_laels,predict)
    print_tab('Concatenation accuracy:%s'%acc, tab_level)

    print_tab("Generating classification report...\n\n\n", tab_level)

    acc=accuracy_score(test_laels,predict)
    pre=precision_score(test_laels,predict,average='weighted')
    re=recall_score(test_laels,predict,average='weighted')
    f1=f1_score(test_laels,predict,average='weighted')
    print('ensemble accuracy: %s'%acc)
    print('precision: %s'%pre)
    print('recall: %s'%re)
    print('f1: %s'%f1)

    print(confusion_matrix(test_laels, predict))
    target_names = [f'{i}' for i in range(numclass)]
    print(classification_report(test_laels, predict, target_names=target_names))

    _, label_map = get_label_map(numclass)

    y_true_list = [int(i) for i in test_laels]
    y_pred_list = [int(i) for i in predict]

    y_true_mapped = [label_map[y] for y in y_true_list]
    y_pred_mapped = [label_map[y] for y in y_pred_list]


    print(classification_report(y_true_mapped, y_pred_mapped, zero_division=1))
    print('\n')
    print('                    Confusion Matrix')



def test_on_shuffled_on_ensemble(numclass):

    tab_level = 1

    TARGET_SIZE=(224,224)     
    INPUT_SIZE=(224,224,3)    
    BATCHSIZE=32          

    test_datagen = ImageDataGenerator(rescale=1./255)

            
    validation_generator = test_datagen.flow_from_directory(
                f'./{numclass}class_test_224_shuffled/',
                target_size=TARGET_SIZE,
                batch_size=BATCHSIZE,
                class_mode='categorical')
    
    label=validation_generator.class_indices
    label={v: k for k, v in label.items()}
    
    ensemble_model=load_model(f'./{numclass}class_ensemble.keras')

    #read images from validation folder


    rootdir = f'./{numclass}class_test_224_shuffled/'

    test_laels = []
    test_images=[]
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if not (file.endswith(".jpeg"))|(file.endswith(".jpg"))|(file.endswith(".png")):
                continue
            test_laels.append(subdir.split('/')[-1])
            test_images.append(os.path.join(subdir, file))



    #test the averaging model on the validation set

    predict=[]
    length=len(test_images)
    t1 = time.time()
    for i in range((length//127)+1):
        inputimg=test_images[127*i:127*(i+1)]
        test_batch=[]
        for path in inputimg:
            thisimg=np.array(Image.open(path))/255
            test_batch.append(thisimg)
        #print(i, np.array(test_batch).shape)
        ensemble_model_batch=ensemble_model.predict(np.array(test_batch))
        predict_batch=list(np.argmax(ensemble_model_batch,axis=1))
        predict_batch=[label[con] for con in predict_batch]
        predict.append(predict_batch)

    predict=sum(predict,[])

    t2 = time.time()
    print_tab('The testing time is :%f seconds' % (t2-t1), tab_level)

    acc=accuracy_score(test_laels,predict)
    print_tab('Concatenation accuracy:%s'%acc, tab_level)

    print_tab("Generating classification report...\n\n\n", tab_level)

    acc=accuracy_score(test_laels,predict)
    pre=precision_score(test_laels,predict,average='weighted')
    re=recall_score(test_laels,predict,average='weighted')
    f1=f1_score(test_laels,predict,average='weighted')
    print('ensemble accuracy: %s'%acc)
    print('precision: %s'%pre)
    print('recall: %s'%re)
    print('f1: %s'%f1)

    print(confusion_matrix(test_laels, predict))
    target_names = [f'{i}' for i in range(numclass)]
    print(classification_report(test_laels, predict, target_names=target_names))

    _, label_map = get_label_map(numclass)

    y_true_list = [int(i) for i in test_laels]
    y_pred_list = [int(i) for i in predict]

    y_true_mapped = [label_map[y] for y in y_true_list]
    y_pred_mapped = [label_map[y] for y in y_pred_list]


    print(classification_report(y_true_mapped, y_pred_mapped, zero_division=1))
    print('\n')
    print('                    Confusion Matrix')





if __name__ == "__main__":
    
    total_start_time = time.time()
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--numclass", type=int, default=8, help="The number of target classes in the data")
    args = parser.parse_args()

    # Requires Python 3.11 (python version 3.11.13 is used)
    generate_images(args.numclass)
    # training_and_testing(args.numclass, train_constituent_models=True)
    test_on_shuffled_on_ensemble(args.numclass)

    
    total_end_time = time.time()

    print(f'Total elapsed time: {total_end_time-total_start_time} seconds')