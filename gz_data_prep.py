import os
import numpy as np
import zipfile
import pandas as pd


zipfile_name_train_x = 'images_training_rev1.zip'
zipfile_name_train_y = 'training_solutions_rev1.zip'
zipfile_name_ds2_x = 'ds2_x.zip'

num_of_train_expls = 10
threshold = 0.66

#ZIP TRAINING Y FILE EXTRACTION - image info
zip_data = zipfile.ZipFile(zipfile_name_train_y, 'r')
zip_data.extractall()

#PREPARING DATA SET 2 --- THRESHOLD USED TO DETERMINE THE CLASS(HENCE SOME CHALLENGING EXAMPLES [0.38,0.33, 0.29] ARE THROWN AWAY)
training_y  = pd.read_csv(os.path.join(os.getcwd(), zipfile_name_train_y[0:-3]+'csv'))#reading csv file into pandas
training_y =  training_y.iloc[:,0:4]#extracting first 3 columns(not counting gal id)

#applying threshold and making one hot end vectorization
training_y['Class1.1'] = np.where(training_y['Class1.1'] >= threshold, 1, 0)
training_y['Class1.2'] = np.where(training_y['Class1.2'] >= threshold, 1, 0)
training_y['Class1.3'] = np.where(training_y['Class1.3'] >= threshold, 1, 0)

#throwing away empty rows &  where Class1.3 == 1
list_of_dropouts = training_y.loc[(training_y['Class1.1']==0) & (training_y['Class1.2'] ==0) | (training_y['Class1.3'] ==1)] #vector of galaxyids of examples that we need to drop
training_y = training_y.loc[(training_y['Class1.1']==1) | (training_y['Class1.2'] ==1) & (training_y['Class1.3'] ==0)]
training_y = training_y.iloc[:,0:3]
training_y.to_csv("y_training_ds2.csv", sep='\t', encoding='utf-8')

#zip training X file extraction
zip_data = zipfile.ZipFile(zipfile_name_train_x, 'r')
name_list = zip_data.namelist()

if num_of_train_expls != True:#if you want just a part of training data change num_of_train_expls else assign it True
    for i in name_list[1:(num_of_train_expls+1)]:
        if int(i[21:-4]) not in np.transpose(np.array(list_of_dropouts))[0]:
            zip_data.extract(i)
else:
    for i in name_list[1:]:
        if int(i[21:-4]) not in np.transpose(np.array(list_of_dropouts))[0]:
            zip_data.extract(i)
os.rename(os.path.join(os.getcwd(), 'images_training_rev1'), os.path.join(os.getcwd(), 'x_training_ds2'))
zip_data.close()






#PREPARING DATA SET 1 --- original dataset with excluded class3=True - not galaxies
training_y  = pd.read_csv(os.path.join(os.getcwd(), zipfile_name_train_y[0:-3]+'csv'))#reading csv file into pandas
training_y =  training_y.iloc[:,0:4]#extracting first 3 columns(not counting gal id)

#applying maximization and making one hot end vectorization
training_y['max_value'] = training_y.iloc[:,1:4].max(axis=1)
print(training_y.head())
training_y['Class1.1'] = np.where(training_y['Class1.1'] >= training_y['max_value'], 1, 0)
training_y['Class1.2'] = np.where(training_y['Class1.2'] >= training_y['max_value'], 1, 0)
training_y['Class1.3'] = np.where(training_y['Class1.3'] >= training_y['max_value'], 1, 0)
training_y =  training_y.iloc[:,0:4]

#throw out examples with Class1.3=1
list_of_dropouts = training_y.loc[(training_y['Class1.1']==0) & (training_y['Class1.2'] ==0) | (training_y['Class1.3'] ==1)]
print(list_of_dropouts.head())
training_y = training_y.loc[(training_y['Class1.1']==1) | (training_y['Class1.2'] ==1) & (training_y['Class1.3'] ==0)]
training_y = training_y.iloc[:,0:3]
training_y.to_csv("y_training_ds1.csv", sep='\t', encoding='utf-8')

#zip training X file extraction for dataset 1
zip_data = zipfile.ZipFile(zipfile_name_train_x, 'r')
name_list = zip_data.namelist()

if num_of_train_expls != True:#if you want just a part of training data change num_of_train_expls else assign it True
    for i in name_list[1:(num_of_train_expls+1)]:
        if int(i[21:-4]) not in np.transpose(np.array(list_of_dropouts))[0]:
            zip_data.extract(i)
else:
    for i in name_list[1:]:
        if int(i[21:-4]) not in np.transpose(np.array(list_of_dropouts))[0]:
            zip_data.extract(i)
os.rename(os.path.join(os.getcwd(), 'images_training_rev1'), os.path.join(os.getcwd(), 'x_training_ds1'))
zip_data.close()
