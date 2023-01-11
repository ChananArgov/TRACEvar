"""The Aim of this script is prioritize variants using TRACEvar model
Input:
1. CADD annotation file (patient_cadd_path).
2. Relevant tissue name (tissue1, see tissue_dict for options).
3. job_name.
"""

import os
import pickle
import sys
import pandas as pd

pd.options.mode.chained_assignment = None

"---------------------------- Load Data ------------------------------"
job_name = 'Example_1'

patient_cadd_path = os.path.join('..', 'Example', '2023-01-08__12_48_25_cadd.tsv')
Patient_CADD_data = pd.read_csv(patient_cadd_path, sep='\t', skiprows=1)

Patient_CADD_data = Patient_CADD_data.rename(columns={'GeneID': 'GeneID_y', '#Chrom': '#Chr'})
Patient_CADD_data['#Chr'] = Patient_CADD_data['#Chr'].astype(str)
annotypes = ['CodingTranscript', 'Transcript', 'RegulatoryFeature', 'Intergenic', 'NonCodingTranscript']
Patient_CADD_data = Patient_CADD_data.sort_values('AnnoType', key=lambda s: s.apply(annotypes.index), ignore_index=True)
Patient_CADD_data = Patient_CADD_data.drop_duplicates(subset=['#Chr', 'Pos', 'Ref', 'Alt'], keep='first')

trace_features_path = '../Data/df_complete_dataset.csv'
TRACE_data = pd.read_csv(trace_features_path)

TRACE_data = TRACE_data.rename(columns={'Unnamed: 0': 'GeneID_y'})

tissue_dict = {'brain': 'brain', 'Heart': 'Heart - Left Ventricle', 'Kidney': 'kidney', 'Muscle-Skeletal':'Muscle - Skeletal', 'Skin':'Skin - Not Sun Exposed', 'Liver':'Liver', 'Nerve':'Nerve - Tibial', 'Blood': 'Whole Blood', 'brain-0':'brain-0', 'brain-1':'brain-1','brain-2':'brain-2', 'brain-3':'brain-3', 'Testis': 'Testis', 'Ovary': 'Ovary', 'Pituitary':'Pituitary', 'Lung':'Lung', 'Artery-Aorta':'Artery - Aorta'}
tissue1 = 'brain'
tissue = tissue_dict[tissue1]


features_model_path = '../Trained Models/' + tissue + "_Features_dict.pkl"
with open(features_model_path, 'rb') as handle:
    model_features_dict = pickle.load(handle)

relevant_model_path = '../Trained Models/' + tissue + '_RF_Model.pkl'
print(relevant_model_path)
with open(relevant_model_path, 'rb') as handle:
    model = pickle.load(handle)

"-------------------------- Relevant Columns --------------------------"

path = '../Data/Relevant_Columns_Names_Edited_2.csv'
Relevant_Cols_df = pd.read_csv(path)
overlap_cols = Relevant_Cols_df['Feature'].tolist()
print(overlap_cols)
rename_dict = dict(zip(overlap_cols, Relevant_Cols_df['Feature Name'].tolist()))
# print(rename_dict)

feature_order_path = '../Trained Models/' + tissue + '_Features_Order.csv'
Feature_Order = pd.read_csv(feature_order_path)
print('Feature_Order')
print(Feature_Order)
feature_order_list = Feature_Order['0'].to_list()

"-------------------------- Data PreProcessing ------------------------"


def edit_trace_data(TRACE_data):
    
    trace_features = list(TRACE_data)
    relevant_trace_features = [x for x in trace_features if "_causal" not in x]
    Relevant_TRACE = TRACE_data[relevant_trace_features]
    Relevant_TRACE = Relevant_TRACE[Relevant_TRACE.filter(regex='^((?!embedding).)*$').columns]
    Relevant_TRACE = Relevant_TRACE.fillna(0)
    return Relevant_TRACE


def edit_cadd_data(Relevant_Data):

    one_hot_columns = ['Type', 'AnnoType', 'Consequence', 'Domain', 'Dst2SplType']
    one_hot = pd.get_dummies(Relevant_Data[one_hot_columns])
    one_hot_columns = [c for c in one_hot_columns if c != 'Type']
    Relevant_Data = Relevant_Data.drop(one_hot_columns, axis=1)
    Relevant_Data = Relevant_Data.join(one_hot)
    print('relevant_data_1: ', Relevant_Data)

    "---------------------- Missing Values Imputation ---------------"
    
    special_imputation_cols = {'SIFTval':1, 'GC':0.42, 'CpG':0.02, 'priPhCons':0.115, 'mamPhCons':0.079, 'verPhCons':0.094,'priPhyloP':-0.033, 'mamPhyloP':-0.038, 'verPhyloP':0.017, 'GerpN':1.91, 'GerpS':-0.2}
    
    for cl in special_imputation_cols:
        Relevant_Data[cl] = Relevant_Data[cl].fillna(special_imputation_cols[cl])
        
    Relevant_Data.fillna(0, inplace=True)

    return Relevant_Data


def merge_trace_and_cadd_data(Edited_CADD_data, Relevant_TRACE):
    Model_input = pd.merge(Edited_CADD_data, Relevant_TRACE, how='inner', on='GeneID_y')
    return Model_input


Relevant_TRACE = edit_trace_data(TRACE_data)
print(Relevant_TRACE)
non_relevant_patient = ['ConsDetail', 'motifEName', 'FeatureID', 'CCDS', 'Intron', 'Exon',
                        'SIFTcat', 'PolyPhenCat', 'bStatistic', 'targetScan', 'dbscSNV-rf_score', 'oAA',  'nAA', 'Segway']  # it will be good to replace oAA and nAA with blssuom64 matrix. What bStatistic doing?relevant_cols = [c for c in list(Patient_CADD_data) if c not in non_relevant_patient]
relevant_cols = [c for c in list(Patient_CADD_data) if c not in non_relevant_patient]
Edited_CADD_data = edit_cadd_data(Patient_CADD_data[relevant_cols])
print(Edited_CADD_data)

Model_input = merge_trace_and_cadd_data(Edited_CADD_data, Relevant_TRACE)
Model_input.rename(columns=rename_dict, inplace=True)

"--------------------- Deal with Missed Features ----------------------------------"

model_features = [*model_features_dict]
missed_features_in_patient = [x for x in model_features if x not in list(Model_input)]
print('missed_features_in_patient', missed_features_in_patient)
missed_features_in_model = [x for x in list(Model_input) if x not in model_features]
print('missed_features_in_model', missed_features_in_model)
for missed_f in missed_features_in_patient:
    Model_input[missed_f] = model_features_dict[missed_f]

relevant_input_cols_2 = [f for f in list(Model_input) if f not in missed_features_in_model]

"--------------------- Prioritize Patient Variants ----------------------------------"

patient_predict_proba = model.predict_proba(Model_input[feature_order_list])  # [common_features] relevant_input_cols_2
patient_predictions = patient_predict_proba[:, 1]
prediction_df = pd.DataFrame(patient_predictions, columns=['Pathological_probability'])
Results_df = pd.concat([Model_input, prediction_df], axis=1)

result_cols = ['GeneName', 'GeneID_y', '#Chr', 'Pos', 'Ref', 'Alt', 'Type', 'Length', 'SIFTval', 'PolyPhenVal', 'PHRED', 'Pathological_probability']
result_cols_new = []

for c in result_cols:
   if c in rename_dict:
      result_cols_new.append(rename_dict[c])
   else: result_cols_new.append(c)
print(result_cols_new)
print(list(Results_df))

Relevant_Results = Results_df[result_cols_new]

print(Model_input)
print(Relevant_Results)
Relevant_Results = Relevant_Results.sort_values('Pathological_probability', ascending=False)
print(Relevant_Results)

out_put_path = os.path.join('../Output/', job_name + '_' + tissue1 + '_2.csv')
Relevant_Results.to_csv(out_put_path, index=False)
