"---------------------- Imports -----------------------------------------"

import matplotlib.pyplot as plt
# import seaborn as sns
import os
import pandas as pd
import sys
# import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap  # package used to calculate Shap values
pd.options.mode.chained_assignment = None
#import pickle
import pickle as pickle
from datetime import datetime
start=datetime.now()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import statistics

"----------------------------- Load Data -----------------------------------"
file_name = sys.argv[1]
short_name = file_name.split('/')[-1]
time_id = short_name.replace('_cadd.csv', '') 

Patient_CADD_data = pd.read_csv(file_name)
TRACE_data = pd.read_csv(sys.argv[2])
TRACE_data = TRACE_data.rename(columns={'Unnamed: 0': 'GeneID_y'})

relevant_variant_index = int(sys.argv[3])
tissue = sys.argv[4]
prediction_output_path = sys.argv[5]
Prediction_Output = pd.read_csv(prediction_output_path)
path = '/storage16/users/chanana/PathoSearch/ML-Scripts'
genome_verssion = sys.argv[6]
timestamp = sys.argv[7]

#relevant_model_path = path + '/Prediction_Models_'+ genome_verssion + '/'  + tissue + '_RF_Model.pkl'
relevant_model_path = '/storage16/users/chanana/PathoSearch/ML-Scripts/Create_Models_Scripts/Estimate_Models_Analysis/Prediction_Models_' + genome_verssion + '_Plot/'  + tissue + '_RF_Model.pkl'
print(relevant_model_path)
with open(relevant_model_path, 'rb') as handle:
    model = pickle.load(handle)

#features_model_path = path + '/Prediction_Models_'+ genome_verssion + '/' + tissue + '_Features_dict.pkl'
features_model_path =  '/storage16/users/chanana/PathoSearch/ML-Scripts/Create_Models_Scripts/Estimate_Models_Analysis/Prediction_Models_' + genome_verssion + '_Plot/'  + tissue + '_Features_dict.pkl'
print(features_model_path)

with open(features_model_path, 'rb') as handle:
    model_features_dict = pickle.load(handle)

feature_order_path = path + '/Prediction_Models_'+ genome_verssion + '/'  + tissue + '_RF_Model.pkl'
feature_order_path = '/storage16/users/chanana/PathoSearch/ML-Scripts/Create_Models_Scripts/Estimate_Models_Analysis/SHAP_Explainers_' + genome_verssion + '_Plot/'  + tissue + '_Features_Order.csv'
Feature_Order = pd.read_csv(feature_order_path)
print('Feature_Order')
print(Feature_Order)
feature_order_list = Feature_Order['0'].to_list()


path = '/storage16/users/chanana/PathoSearch/ML-Scripts/Features_Group_Clasification.csv'
Features_Group_Clasification = pd.read_csv(path)
print(Features_Group_Clasification)


"-------------------------- Data PreProcessing Functions ------------------------"


def edit_trace_data(TRACE_data):

    trace_features = list(TRACE_data)
    relevant_trace_features = [x for x in trace_features if "_causal" not in x]
    Relevant_TRACE = TRACE_data[relevant_trace_features]
    Relevant_TRACE = Relevant_TRACE[Relevant_TRACE.filter(regex='^((?!embedding).)*$').columns]
    Relevant_TRACE = Relevant_TRACE.fillna(0)
    return Relevant_TRACE


def edit_cadd_data(Relevant_Data):
    cols = list(Relevant_Data)
    #print('Relevant_Data', cols)
    one_hot_columns = ['Type', 'AnnoType', 'Consequence', 'Domain', 'Dst2SplType']  # , 'EnsembleRegulatoryFeature'
    # Get one hot encoding of columns B
    one_hot = pd.get_dummies(Relevant_Data[one_hot_columns])
    # Drop column B as it is now encoded
    one_hot_columns = [c for c in one_hot_columns if c != 'Type']
    Relevant_Data = Relevant_Data.drop(one_hot_columns, axis=1)
    # Join the encoded df
    Relevant_Data = Relevant_Data.join(one_hot)
    #print('relevant_data_1: ', Relevant_Data)
    cHmm_columns = Relevant_Data.columns[Relevant_Data.columns.str.contains(pat='cHmm_E')].tolist()
    fill_zero_columns = ['motifECount', 'motifEHIPos', 'motifEScoreChng', 'mirSVR-Score', 'mirSVR-E', 'mirSVR-Aln',
                         'tOverlapMotifs', 'motifDist'] + cHmm_columns  # motifs with high number of nan 97%
    Relevant_Data[fill_zero_columns] = Relevant_Data[fill_zero_columns].fillna(value=0)
    fill_common_columns = ['cDNApos', 'relcDNApos', 'CDSpos', 'relCDSpos', 'protPos', 'relProtPos', 'Dst2Splice',
                           'SIFTval', 'PolyPhenVal', 'GerpRS', 'GerpRSpval', 'GerpN', 'GerpS', 'all Enc',
                           'Grantham', 'All SpliceAI', 'All MMSp', 'Dist2Mutation', 'All 00bp', 'dbscSNV',
                           'RemapOverlapTF', 'RemapOverlapCL', 'Trace Features']  # Locations, is this right?
    for cl in list(Relevant_Data):
        # print(cl)
        # print(Relevant_Data[cl])
        try:
            Relevant_Data[cl] = Relevant_Data[cl].fillna(Relevant_Data[cl].value_counts().idxmax())
        except:
            Relevant_Data[cl] = Relevant_Data[cl].fillna(0)
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
#path12 = '/storage16/users/chanana/PathoSearch/ML-Scripts/Saved_Model_Input/2021-06-21__11_49_10_ModelInput.csv'
#Model_input.to_csv(path12)
" ---------------------- relevant variant original ind ---------------------"
Relevant_Variant = Prediction_Output.iloc[relevant_variant_index]
print('Relevant_Variant')
print(Relevant_Variant)

variant_gene = Relevant_Variant['GeneID_y']
variant_pos = Relevant_Variant['Pos']
variant_type = Relevant_Variant['Type']
variant_gene_name = Relevant_Variant['GeneName']
variant_chr = Relevant_Variant['#Chr']

original_variant_index = Model_input[(Model_input['GeneID_y'] == variant_gene) & (Model_input['Pos']==variant_pos)].index.values[0]
print('original_variant_index', original_variant_index)


"--------------------- Deal with Missed Features ----------------------------------"

model_features = [*model_features_dict]
#model_features = [model_features_dict]
#print(model_features)
missed_features_in_patient = [x for x in model_features if x not in list(Model_input)]
print('missed_features_in_patient', missed_features_in_patient)
missed_features_in_model = [x for x in list(Model_input) if x not in model_features]
#print('missed_features_in_model', missed_features_in_model)
for missed_f in missed_features_in_patient:
    Model_input[missed_f] = model_features_dict[missed_f]
relevant_input_cols_2 = [f for f in list(Model_input) if f not in missed_features_in_model]

"------------------------------ SHAP Issue ----------------------------------"

#Model_Input_Data = Model_input[relevant_input_cols_2]
Model_Input_Data = Model_input[feature_order_list]


print(Model_Input_Data)

#path = '/storage16/users/chanana/PathoSearch/ML-Scripts/SHAP_Explainers_'+ genome_verssion + '/' + tissue + '_RF_Explainer.pkl'
path ='/storage16/users/chanana/PathoSearch/ML-Scripts/Create_Models_Scripts/Estimate_Models_Analysis/SHAP_Explainers_' + genome_verssion + '_Plot/'  + tissue + '_RF_Explainer.pkl'
print(path)
with open(path, 'rb') as handle:
   explainerModel = pickle.load(handle)

shap_values_Model = explainerModel.shap_values(Model_Input_Data.iloc[original_variant_index])

short_name = variant_gene_name + ' ' + variant_chr + ' ' + str(variant_pos)
print(shap_values_Model)
print(explainerModel.expected_value)


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
# plt.yticks(fontsize=14)#, rotation=90
# plt.tight_layout()
plot_title = short_name + ' ' + tissue.strip()

p = 0.08  # Probability 0.4
new_base_value = np.log(p / (1 - p))  # the logit function
shap.decision_plot(explainerModel.expected_value[1], shap_values_Model[1], Model_Input_Data.iloc[[original_variant_index]], show=False, highlight=0)  # Rf : explainer.expected_value[1],link='logit', new_base_value=new_base_value
plt.title(plot_title + ' SHAP Decision Plot', x=0.5, y=1.1)
# plt.title("title", x=0.9, y=0.9)
all_axes = fig.get_axes()
ax = all_axes[0]
# ax.tick_params(axis="y", labelsize=3, direction='in')
ax.tick_params(axis="y", direction="in", labelsize=9, pad=-190)  # , pad=-10
ax.tick_params(axis="y", labelsize=9)
plt.margins(1, tight=True)

path = '/storage16/users/chanana/PathoSearch/ML-Scripts'
file_name = timestamp + "_" + tissue + "_decision_plot.jpg"
out_path = os.path.join(path, 'SHAP_Output', file_name)
plt.savefig(out_path, dpi=100)
plt.close()

print('Runing time: ',datetime.now()-start)

"----------------------------------------------- Group Plot Script ----------------------------------"
def shap_group_value_calculation(tissue, Shap_Values):

    tissue_word = tissue.split(' ')[0]
    if 'brain' in tissue:
        tissue_word = 'Brain'
    elif 'Whole' in tissue:
        tissue_word = 'Blood'
    elif 'kidney' in tissue:
        tissue_word = 'Kidney'

    trace_groups = ['Absolute Expression_Median_SHAP', 'Preferential Expression_Median_SHAP', 'Process Activity_Median_SHAP', 'Differential PPIs_Median_SHAP', 'PPIs_Median_SHAP', 'Paralog Relationships_Median_SHAP', 'eQTL_Median_SHAP', 'Expression Variability_Median_SHAP', 'Expression during Development_Median_SHAP', 'Expression variability during Development_Median_SHAP']
    feature_groups = Features_Group_Clasification['Features_Group'].unique()

    for group in feature_groups:
        print('@', group)
        features = Features_Group_Clasification['col_name'][Features_Group_Clasification['Features_Group'] == group].tolist()
        if group+'_Median_SHAP' in trace_groups:
            features = Features_Group_Clasification['col_name'][(Features_Group_Clasification['Features_Group'] == group)&(Features_Group_Clasification['col_name'].str.contains('|'.join([tissue_word, tissue_word.capitalize(), tissue_word.lower()])))].tolist()
        relevant_features = [f for f in features if f in list(Shap_Values)]
        print('relevant_features', relevant_features)
        Shap_Values[group + '_Median_SHAP'] = Shap_Values[relevant_features].median(axis=1)
        print(features)
        print(Shap_Values[group + '_Median_SHAP'])

plt.rcParams["figure.figsize"] = (12,35)
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = True

path = '/storage16/users/chanana/PathoSearch/ML-Scripts/Create_Models_Scripts/Estimate_Models_Analysis/SHAP_Explainers_GRCh37_Plot/'+ tissue + '_Shap_Group_Values_Tissue_Specific.csv'
Shap_Groups = pd.read_csv(path)
Shap_Groups.rename(columns={'Unnamed: 0':'VariationID'}, inplace=True)
print(Shap_Groups)


groups_list = [f for f in list(Shap_Groups) if '_Median_SHAP' in f]
print(groups_list)
groups_df_list = []
for g in groups_list:
    group_name = g.replace('_Median_SHAP', '')
    Long_Shap_Group = pd.DataFrame(Shap_Groups[g].tolist(),columns =['Median_SHAP_Value'])
    Long_Shap_Group['Group'] = group_name
    Long_Shap_Group['Is_pathogenic'] = Shap_Groups['Is_pathogenic'].tolist()
    groups_df_list.append(Long_Shap_Group)

All_long = pd.concat(groups_df_list)
print(All_long)
groups = All_long['Group'].unique()
print(groups)

Shap_Values = pd.DataFrame([shap_values_Model[1]], columns=feature_order_list)
#Shap_Values.columns = feature_order_list

shap_group_value_calculation(tissue, Shap_Values)

fig, axs = plt.subplots(len(groups))
c = 0
for g in groups:

    print('@', c,  g)
    patient_value = Shap_Values[g + '_Median_SHAP'].values[0]
    print(patient_value)
    
    axs[c].set_yscale('symlog')#"linear", "log", "symlog", "logit"
    sns.kdeplot(data=All_long[(All_long['Group'] == g)&(All_long['Is_pathogenic'] == False)], x='Median_SHAP_Value', ax = axs[c], color='blue', bw=.2, fill=True)
    sns.kdeplot(data=All_long[(All_long['Group'] == g)&(All_long['Is_pathogenic'] == True)], x='Median_SHAP_Value', ax = axs[c], color='red', bw=.2, fill=True)

    ylim = axs[c].get_ylim()
    print('ylim', ylim)
    y_mean = statistics.mean(ylim)
    print('y_mean', y_mean)
    axs[c].annotate("", xy=(patient_value, 0), xytext=(patient_value, y_mean), arrowprops=dict(arrowstyle="->"))
    axs[c].set_title(g, backgroundcolor= 'yellow')

    c += 1

fig.tight_layout()
path = '/storage16/users/chanana/PathoSearch/ML-Scripts/SHAP_Output'
file_name = timestamp + "_" + tissue + "_shap_output.jpg"
out_path = os.path.join(path, file_name)
plt.savefig(out_path, dpi=100)
plt.close()