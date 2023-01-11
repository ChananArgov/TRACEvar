"""The Aim of this script is to give an interpretation for specific variant prediction using shap package
Input:
1. CADD annotation file (patient_cadd_path).
2. Prioritized variants file obtained by running Prioritize_Variants_By_TRACEvar.py script(prediction_output_path)
3. Relevant tissue name (tissue1, see tissue_dict for options).
4. Relevant variant index in the Prioritized variants file (relevant_variant_index).
5. job_name.
"""

"---------------------- Imports -----------------------------------------"


import pandas as pd
import numpy as np
import shap  # package used to calculate Shap values
pd.options.mode.chained_assignment = None
import pickle as pickle
from datetime import datetime
start=datetime.now()
import matplotlib.pyplot as plt
import pandas as pd
import os

"----------------------------- Load Data -----------------------------------"
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

relevant_variant_index = 0 # The first variant for example

tissue_dict = {'brain': 'brain', 'Heart': 'Heart - Left Ventricle', 'Kidney': 'kidney', 'Muscle-Skeletal':'Muscle - Skeletal', 'Skin':'Skin - Not Sun Exposed', 'Liver':'Liver', 'Nerve':'Nerve - Tibial', 'Blood': 'Whole Blood', 'brain-0':'brain-0', 'brain-1':'brain-1','brain-2':'brain-2', 'brain-3':'brain-3', 'Testis': 'Testis', 'Ovary': 'Ovary', 'Pituitary':'Pituitary', 'Lung':'Lung', 'Artery-Aorta':'Artery - Aorta'}
tissue1 = 'brain' # for example
tissue = tissue_dict[tissue1]

prediction_output_path = os.path.join('../Output/', job_name + '_' + tissue1 + '.csv')
Prediction_Output = pd.read_csv(prediction_output_path)

relevant_model_path = '../Trained Models/' + tissue + '_RF_Model.pkl'
print(relevant_model_path)
with open(relevant_model_path, 'rb') as handle:
    model = pickle.load(handle)

features_model_path = '../Trained Models/' + tissue + "_Features_dict.pkl"
print(features_model_path)
with open(features_model_path, 'rb') as handle:
    model_features_dict = pickle.load(handle)

feature_order_path = '../Trained Models/' + tissue + '_Features_Order.csv'
Feature_Order = pd.read_csv(feature_order_path)
print('Feature_Order')
print(Feature_Order)
feature_order_list = Feature_Order['0'].to_list()

"-------------------------- Relevant Columns --------------------------"
path = '../Data/Relevant_Columns_Names_Edited_2.csv'
Relevant_Cols_df = pd.read_csv(path)
overlap_cols = Relevant_Cols_df['Feature'].tolist()
print(overlap_cols)
rename_dict = dict(zip(overlap_cols, Relevant_Cols_df['Feature Name'].tolist()))


"-------------------------- Data PreProcessing Functions ------------------------"


def edit_trace_data(TRACE_data):

    trace_features = list(TRACE_data)
    relevant_trace_features = [x for x in trace_features if "_causal" not in x]
    Relevant_TRACE = TRACE_data[relevant_trace_features]
    Relevant_TRACE = Relevant_TRACE[Relevant_TRACE.filter(regex='^((?!embedding).)*$').columns]
    Relevant_TRACE = Relevant_TRACE.fillna(0)

    return Relevant_TRACE


def edit_cadd_data(Relevant_Data):

    one_hot_columns = ['Type', 'AnnoType', 'Consequence', 'Domain', 'Dst2SplType']  # , 'EnsembleRegulatoryFeature'
    one_hot = pd.get_dummies(Relevant_Data[one_hot_columns])
    one_hot_columns = [c for c in one_hot_columns if c != 'Type']
    Relevant_Data = Relevant_Data.drop(one_hot_columns, axis=1)
    Relevant_Data = Relevant_Data.join(one_hot)

    "---------------------- Missing Values Imputation ---------------"
    
    special_imputation_cols = {'SIFTval':1, 'GC':0.42, 'CpG':0.02, 'priPhCons':0.115, 'mamPhCons':0.079, 'verPhCons':0.094,'priPhyloP':-0.033, 'mamPhyloP':-0.038, 'verPhyloP':0.017, 'GerpN':1.91, 'GerpS':-0.2}
    
    for cl in special_imputation_cols:
        Relevant_Data[cl] = Relevant_Data[cl].fillna(special_imputation_cols[cl])
        
    Relevant_Data.fillna(0, inplace=True)

    return Relevant_Data


def merge_trace_and_cadd_data(Edited_CADD_data, Relevant_TRACE):
    Model_input = pd.merge(Edited_CADD_data, Relevant_TRACE, how='inner', on='GeneID_y')
    return Model_input
"------------------------ Prossesing Data -----------------------------"

Relevant_TRACE = edit_trace_data(TRACE_data)
print(Relevant_TRACE)
non_relevant_patient = ['ConsDetail', 'motifEName', 'FeatureID', 'CCDS', 'Intron', 'Exon',
                        'SIFTcat', 'PolyPhenCat', 'bStatistic', 'targetScan', 'dbscSNV-rf_score', 'oAA',  'nAA', 'Segway']  # it will be good to replace oAA and nAA with blssuom64 matrix. What bStatistic doing?relevant_cols = [c for c in list(Patient_CADD_data) if c not in non_relevant_patient]
relevant_cols = [c for c in list(Patient_CADD_data) if c not in non_relevant_patient]
Edited_CADD_data = edit_cadd_data(Patient_CADD_data[relevant_cols])
print(Edited_CADD_data)

Model_input = merge_trace_and_cadd_data(Edited_CADD_data, Relevant_TRACE)

" ---------------------- relevant variant original ind ---------------------"
Relevant_Variant = Prediction_Output.iloc[relevant_variant_index]
print('Relevant_Variant')
print(Relevant_Variant)

variant_gene = Relevant_Variant['CADD | GeneID y']
variant_pos = Relevant_Variant['CADD | Pos']
variant_type = Relevant_Variant['CADD | Type']
variant_gene_name = Relevant_Variant['GeneName']
variant_chr = Relevant_Variant['CADD |Chr']
print(list(Model_input)[:10])

original_variant_index = Model_input[(Model_input['GeneID_y'] == variant_gene) & (Model_input['Pos']==variant_pos)].index.values[0]
print('original_variant_index', original_variant_index)


"--------------------- Deal with Missed Features ----------------------------------"

Model_input.rename(columns=rename_dict, inplace=True)
model_features = [*model_features_dict]
missed_features_in_patient = [x for x in model_features if x not in list(Model_input)]
print('missed_features_in_patient', missed_features_in_patient)
missed_features_in_model = [x for x in list(Model_input) if x not in model_features]
for missed_f in missed_features_in_patient:
    Model_input[missed_f] = model_features_dict[missed_f]
relevant_input_cols_2 = [f for f in list(Model_input) if f not in missed_features_in_model]

"------------------------------ SHAP Issue ----------------------------------"

Model_Input_Data = Model_input[feature_order_list]


print(Model_Input_Data)

path = '../Trained Models/' + tissue.strip() + '_RF_Explainer.pkl'
print(path)
with open(path, 'rb') as handle:
   explainerModel = pickle.load(handle)

shap_values_Model = explainerModel.shap_values(Model_Input_Data.iloc[original_variant_index])

short_name = variant_gene_name + ' ' + str(variant_chr) + ' ' + str(variant_pos)
print(shap_values_Model)
print(explainerModel.expected_value)


fig = plt.gcf()
plot_title = short_name + ' ' + tissue.strip()

p = 0.08  # Probability 0.4
new_base_value = np.log(p / (1 - p))
shap.decision_plot(explainerModel.expected_value[1], shap_values_Model[1], Model_Input_Data.iloc[[original_variant_index]], show=False, highlight=0)  # Rf : explainer.expected_value[1],link='logit', new_base_value=new_base_value
plt.title(plot_title + ' SHAP Decision Plot', x=0, y=1.05)
all_axes = fig.get_axes()
ax = all_axes[0]

plt.tight_layout()

file_name = job_name + "_" + tissue1  + "_" + str(relevant_variant_index) + "_shap_output3.jpg"
out_path = '../Output/' + file_name

plt.savefig(out_path, dpi=100)
plt.close()

print('Runing time: ',datetime.now()-start)
