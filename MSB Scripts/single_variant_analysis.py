"""
The code for patient pathogenic variant prioritization, and shap explanation plot for the pathogenic variant (Fig. 3B-D).
Due to privacy concerns, we cannot disclose patient data.
"""

import time
import matplotlib.pyplot as plt
import os
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import shap
pd.options.mode.chained_assignment = None
import pickle
import seaborn as sns


"---------------------------- Load Data ------------------------------"

Metadata = pd.read_csv('path')
Slim_dataset = pd.read_csv('..Data/Full_Slim_Dataset_hg37-v1.6.csv', engine='python')#low_memory=False,
Relevant_Cols_df = pd.read_csv('..Relevant_Columns_Names_Edited_2.csv')
overlap_cols = Relevant_Cols_df['Feature'].tolist()
rename_dict = dict(zip(overlap_cols, Relevant_Cols_df['Feature Name'].tolist()))
# print(rename_dict)
overlap_cols_names  = Relevant_Cols_df['Feature Name'].tolist()

def preprocessing_new(Relevant_Data):
    
    "---------------------- One Hot Columns -------------------------"
    
    one_hot_columns = ['Type', 'AnnoType', 'Consequence', 'Domain', 'Dst2SplType']

    one_hot = pd.get_dummies(Relevant_Data[one_hot_columns])
    Relevant_Data = Relevant_Data.drop(one_hot_columns, axis=1)
    Relevant_Data = Relevant_Data.join(one_hot)
    
    "---------------------- Missing Values Imputation ---------------"
    
    special_imputation_cols = {'SIFTval':1, 'GC':0.42, 'CpG':0.02, 'priPhCons':0.115, 'mamPhCons':0.079, 'verPhCons':0.094,'priPhyloP':-0.033, 'mamPhyloP':-0.038, 'verPhyloP':0.017, 'GerpN':1.91, 'GerpS':-0.2}
    
    for cl in special_imputation_cols:
        Relevant_Data[cl] = Relevant_Data[cl].fillna(special_imputation_cols[cl])
        
    Relevant_Data.fillna(0, inplace=True)
    
    return Relevant_Data
    
y_columns = Slim_dataset.columns[Slim_dataset.columns.str.contains(pat = 'disease_causing')].tolist()
non_relevant_columns = ['VariationID', 'OMIMs', 'Manifested_Tissues', '#Chr', 'Pos', 'ConsDetail', 'motifEName', 'GeneID_y', 'FeatureID', 'GeneName', 'CCDS', 'Intron', 'Exon', 'SIFTcat', 'PolyPhenCat', 'bStatistic', 'targetScan', 'dbscSNV-rf_score', 'oAA', 'Ref', 'nAA', 'Alt', 'Segway']# it will be good to replace oAA and nAA with blssuom64 matrix. What bStatistic doing?
non_relevant_columns = non_relevant_columns + y_columns
non_relevant_patient = ['#Chr', 'Pos', 'ConsDetail', 'motifEName', 'GeneID_y', 'FeatureID', 'GeneName', 'CCDS', 'Intron', 'Exon', 'SIFTcat', 'PolyPhenCat', 'bStatistic', 'targetScan', 'dbscSNV-rf_score', 'oAA', 'Ref', 'nAA', 'Alt', 'Segway']# it will be good to replace oAA and nAA with blssuom64 matrix. What bStatistic doing?

cols = Slim_dataset.columns
relevant_columns = [x for x in cols if x not in non_relevant_columns]


import ast
from time import gmtime, strftime

"-------------------------------- Create Ptient Specific Model --------------------------------------------"
file = '../Usher16032_TRACE_CADD-v1.6_GRCh37.csv'
tissue = 'brain'
Patient_data = pd.read_csv(file)
Patient_data = Patient_data[Patient_data['GeneName'] == 'GPR98']
Patient_Relevant = Patient_data[relevant_columns]
Patient_Relevant = preprocessing_new(Patient_Relevant)

Patient_Relevant.rename(columns=rename_dict, inplace=True)
Model_input = Patient_Relevant
features_model_path = '..' + tissue + "_Features_dict.pkl"
with open(features_model_path, 'rb') as handle:
    model_features_dict = pickle.load(handle)

relevant_model_path = '..' + tissue + '_RF_Model.pkl'
print(relevant_model_path)
with open(relevant_model_path, 'rb') as handle:
    model = pickle.load(handle)

feature_order_path = '..' + tissue + '_Features_Order.csv'
Feature_Order = pd.read_csv(feature_order_path)
feature_order_list = Feature_Order['0'].to_list()

"--------------------- Deal with Missed Features ----------------------------------"

model_features = [*model_features_dict]
#model_features = [model_features_dict]
#print(model_features)
missed_features_in_patient = [x for x in model_features if x not in list(Model_input)]
#print('missed_features_in_patient', missed_features_in_patient)
missed_features_in_model = [x for x in list(Model_input) if x not in model_features]
#print('missed_features_in_model', missed_features_in_model)
for missed_f in missed_features_in_patient:
    Model_input[missed_f] = model_features_dict[missed_f]

relevant_input_cols_2 = [f for f in list(Model_input) if f not in missed_features_in_model]


Model_Input_Data = Model_input[feature_order_list]
explainerModel = shap.TreeExplainer(model)

shap_values_Model = explainerModel.shap_values(Model_Input_Data)

print(shap_values_Model)
print(explainerModel.expected_value)


fig = plt.figure(figsize=(10, 6))
plot_title = 'Usher syndrome, GPR98'

p = 0.4  # Probability 0.08
new_base_value = np.log(p / (1 - p))

shap.decision_plot(explainerModel.expected_value, shap_values_Model, Model_Input_Data, show=False, highlight=0, link='logit')
plt.title(plot_title + ' SHAP Decision Plot', x=0, y=1.05)

# Get the current axis
ax = plt.gca()

# Adjust x-axis limits
plt.xlim(ax.get_xlim()[0], min(ax.get_xlim()[1], 1.0) + 0.5)  # Extend the x-axis by 0.5 units but not beyond x = 1

# Fill the extended area with white color
ax.fill_between([min(ax.get_xlim()[1], 1.0), min(ax.get_xlim()[1], 1.0) + 0.5], ax.get_ylim()[0], ax.get_ylim()[1], color='white')

plt.xticks([0.0, 0.5, 1.0, 1.5])

plt.tight_layout()

file_name = "USHER_GPR98_shap_output3.jpg"
out_path = '..' + file_name

plt.savefig(out_path, dpi=300)
plt.close()
