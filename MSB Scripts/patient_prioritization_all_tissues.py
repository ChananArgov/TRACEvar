"""
The code for patient pathogenic variant prioritization, and shap explanation plot for the pathogenic variant (Fig. EV2).
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


pd.options.mode.chained_assignment = None
import pickle
import seaborn as sns

root_directory = 'Your relevant directory'

"---------------------------- Load Data ------------------------------"
Metadata = pd.read_csv(root_directory, 'HPO_Dataset_Edited.csv')
Slim_dataset = pd.read_csv(root_directory + 'full_dataset_2_stars_2022.csv', index_col=0,engine='python')#low_memory=False,
Relevant_Cols_df = pd.read_csv(root_directory, 'Relevant_Columns_Names_Edited_2.csv')
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


"-------------------------------- Create Ptient Specific Model --------------------------------------------"
patient_files_list = Metadata['Dataset File'].tolist()
print(patient_files_list)
patient_files_list = [x for x in patient_files_list if pd.notnull(x)]

import os
import pandas as pd
from time import strftime, gmtime
import pickle


def patient_variants_prioritization(file):

    short_name = file.split('_')[0]
    disease_gene = Metadata['Gene'][Metadata['Dataset File'] == file].values[0]
    tissues_metadata = Metadata['Tissue_Model'][Metadata['Dataset File'] == file].values[0]
    disease_gene = disease_gene.replace("?", "")

    print('---------------------------', short_name, '------------------------')

    patient_path = root_directory +  file
    
    Patient_data = pd.read_csv(patient_path)
    Patient_Relevant = Patient_data[relevant_columns]
    Patient_Relevant = preprocessing_new(Patient_Relevant)

    patient_genes = Patient_data['GeneID_y'].tolist()
    
    tissues = ['Heart - Left Ventricle_disease_causing', 'brain_disease_causing', 'Lung_disease_causing',
               'Muscle - Skeletal_disease_causing',
               'brain-0_disease_causing', 'Liver_disease_causing',
               'Nerve - Tibial_disease_causing', 'kidney_disease_causing',
               'brain-1_disease_causing', 'Skin - Not Sun Exposed_disease_causing',
                'brain-3_disease_causing',
               'Testis_disease_causing', 'Whole Blood_disease_causing', 'brain-2_disease_causing']
    
    # Create an empty DataFrame to store predictions
    final_results_df = pd.DataFrame()

    for tissue in tissues:
        tissue = tissue.replace("_disease_causing", "")
        Patient_Relevant.rename(columns=rename_dict, inplace=True)
        Model_input = Patient_Relevant
        features_model_path = root_directory  + tissue + "_Features_dict.pkl"
        with open(features_model_path, 'rb') as handle:
            model_features_dict = pickle.load(handle)

        relevant_model_path = root_directory + tissue + '_RF_Model.pkl'
        with open(relevant_model_path, 'rb') as handle:
            model = pickle.load(handle)

        feature_order_path = root_directory + tissue + '_Features_Order.csv'
        Feature_Order = pd.read_csv(feature_order_path)
        feature_order_list = Feature_Order['0'].to_list()
    
        # Deal with Missed Features
        model_features = [*model_features_dict]
        missed_features_in_patient = [x for x in model_features if x not in list(Model_input)]
        missed_features_in_model = [x for x in list(Model_input) if x not in model_features]
        for missed_f in missed_features_in_patient:
            Model_input[missed_f] = model_features_dict[missed_f]

        relevant_input_cols_2 = [f for f in list(Model_input) if f not in missed_features_in_model]

        # Prioritize Patient Variants
        patient_predict_proba = model.predict_proba(Model_input[feature_order_list])
        patient_predictions = patient_predict_proba[:, 1]
        
        # Create a new column for each tissue's prediction
        prediction_column_name = 'Pathological_probability_' + tissue
        final_results_df[prediction_column_name] = patient_predictions

    # Include the desired columns from the original data
    desired_columns = ['GeneName', 'GeneID_y', '#Chr', 'Pos', 'Ref', 'Alt', 'Type', 'Length', 'SIFTval', 'PolyPhenVal', 'PHRED']
    final_results_df = pd.concat([Patient_data[desired_columns], final_results_df], axis=1)

    # Add the 'Is_pathogenic' column
    final_results_df['Is_pathogenic'] = False
    final_results_df.loc[final_results_df.GeneName == disease_gene, 'Is_pathogenic'] = True
    final_results_df['Tissues'] = tissues_metadata

    # Save the final DataFrame to a CSV file
    out_put_path = os.path.join(root_directory, short_name + '_predictions.csv')
    final_results_df.to_csv(out_put_path, index=False)

    return short_name, strftime("%Y-%m-%d %H:%M:%S", gmtime())

"-------------------------------- Multiprocessing ----------------------------------"


import multiprocessing as mp

def driver_func_shap():
    PROCESSES = 17
    df_list = []

    with mp.Pool(PROCESSES) as pool:
        results = [pool.apply_async(patient_variants_prioritization(file, )) for file in patient_files_list]
        for r in results:
            results_tuple = r.get(timeout=None)
            print('@', results_tuple[0], ' finished', results_tuple[1])


if __name__ == '__main__':
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    driver_func_shap()
