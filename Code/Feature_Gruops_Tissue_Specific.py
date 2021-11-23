import os
import pandas as pd
import collections
import seaborn as sns
# import  matplotlib
import matplotlib.pyplot as plt
import scipy.stats as sts
" ------------------------- Fig Parameters -------------------------------"

plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams.update({'font.size': 22})

"---------------------------- Load Model Input Data ---------------------------"

path = os.path.join('../..', '..', 'Webtool_Files/Data/CardiomyopathyOtB0551_Model_Input.csv')
Model_Input = pd.read_csv(path)
relevant_cols = [c for c in list(Model_Input) if c != 'Unnamed: 0']
print(Model_Input)

" ------------------------ Classify Features Into Concept Groups: trace = tissue-based features, cadd = genetic, variant-oriented features -----------------"

print('relevant_cols', len(relevant_cols), relevant_cols)
last_col = 'Testis.Senior_Development'
first_col = 'Adipose - Subcutaneous_expression'
trace_cols = relevant_cols[relevant_cols.index(first_col):relevant_cols.index(last_col)+1]
cadd_cols = [c for c in relevant_cols if c not in trace_cols]

print('trace_cols', len(trace_cols), trace_cols)
print('cadd_cols', len(cadd_cols), cadd_cols)

cadd_locations = ['cDNApos', 'relcDNApos', 'CDSpos', 'relCDSpos', 'protPos', 'relProtPos',  'minDistTSS', 'AnnoType_CodingTranscript', 'AnnoType_Intergenic', 'AnnoType_NonCodingTranscript', 'AnnoType_Transcript', 'Consequence_3PRIME_UTR', 'Consequence_5PRIME_UTR']
cadd_mutation_types = ['Length', 'Type_DEL', 'Type_INS', 'Type_SNV']
cadd_consequence = ['Consequence_DOWNSTREAM', 'Consequence_FRAME_SHIFT', 'Consequence_INFRAME','Consequence_INTRONIC', 'Consequence_NONCODING_CHANGE', 'Consequence_NON_SYNONYMOUS',  'Consequence_STOP_LOST', 'Consequence_SYNONYMOUS']
cadd_domains = ['Domain_hmmpanther', 'Domain_lcompl', 'Domain_ncoils', 'Domain_ndomain', 'Domain_sigp']
cadd_scores = ['RawScore', 'PHRED','SIFTval', 'PolyPhenVal', 'ConsScore']
cadd_mir = ['mirSVR-Score', 'mirSVR-E', 'mirSVR-Aln']
cadd_conservation = ['priPhCons', 'mamPhCons', 'verPhCons', 'priPhyloP', 'mamPhyloP', 'verPhyloP', 'GerpRS', 'GerpRSpval', 'GerpN', 'GerpS', 'Grantham']
cadd_motif = ['motifECount', 'motifEHIPos', 'motifEScoreChng', 'motifDist', 'minDistTSE']
cadd_cHmm = [c for c in relevant_cols if 'cHmm' in c]
cadd_freq = ['Freq100bp', 'Rare100bp', 'Sngl100bp', 'Freq1000bp', 'Rare1000bp', 'Sngl1000bp', 'Freq10000bp', 'Rare10000bp', 'Sngl10000bp', 'Dist2Mutation']
cadd_encode = [c for c in relevant_cols if 'Enc' in c]
cadd_TF = ['TFBS', 'TFBSPeaks', 'TFBSPeaksMax', 'tOverlapMotifs']
cadd_splice = ['MMSp_acceptorIntron', 'MMSp_acceptor', 'MMSp_exon', 'MMSp_donor', 'MMSp_donorIntron', 'Consequence_SPLICE_SITE', 'Dst2Splice', 'SpliceAI-acc-gain', 'SpliceAI-acc-loss', 'SpliceAI-don-gain', 'SpliceAI-don-loss', 'Dst2SplType_ACCEPTOR', 'Dst2SplType_DONOR', 'Consequence_CANONICAL_SPLICE', 'dbscSNV-ada_score']
cadd_gc = ['GC', 'CpG']

print(cadd_encode)


classified_features = cadd_encode + cadd_TF + cadd_cHmm + cadd_scores + cadd_freq + cadd_mir + cadd_domains + cadd_motif + cadd_locations + cadd_conservation + cadd_mutation_types + cadd_splice  + cadd_gc + cadd_consequence
print('classified_features', len(classified_features))
non_class = [c for c in cadd_cols if c not in classified_features]
print(non_class)
over_class = [item for item, count in collections.Counter(classified_features).items() if count > 1]
print('over_class', over_class)

trace_expression = [c for c in trace_cols if ('expression' in c) & ('_preferential' not in c)&('Development' not in c)]
trace_pref_exp = [c for c in trace_cols if '_preferential' in c]
trace_pathways = [c for c in trace_cols if '_tipa_pathways' in c]
trace_diffnet =  [c for c in trace_cols if '_diff_net' in c]
trace_interactions =  [c for c in trace_cols if ('_interactors' in c)|('interactions' in c)]
trace_paralogs = [c for c in trace_cols if 'paralogs_ratio' in c]
trace_development = [c for c in trace_cols if ('Development' in c)&('_Development_CV' not in c)]
trace_egen = [c for c in trace_cols if 'egene' in c]
trace_divercity = [c for c in trace_cols if 'LCV' in c]
trace_development_variability = [c for c in trace_cols if '_Development_CV' in c]
classified_trace = trace_diffnet + trace_paralogs + trace_pref_exp + trace_expression + trace_interactions + trace_pathways + trace_development + trace_egen + trace_divercity
non_class_trace = [c for c in trace_cols if c not in classified_trace]
print('non_class_trace', len(non_class_trace), non_class_trace)


over_class_trace = [item for item, count in collections.Counter(classified_trace).items() if count > 1]
print('over_class_trace', over_class_trace)

feature_gruops_dict = {'Absolute Expression': trace_expression, 'Preferential Expression':trace_pref_exp, 'Process Activity':trace_pathways, 'Differential PPIs':trace_diffnet, 'PPIs':trace_interactions, 'Paralog Relationships':trace_paralogs, 'eQTL': trace_egen, 'Expression Variability':trace_divercity,'Expression during Development':trace_development, 'Expression variability during Development':trace_development_variability,
                       'Conservation, Evolution':cadd_conservation, 'Variation Frequency':cadd_freq, 'Variation Consequence': cadd_consequence, 'Pathogenicity Scores':cadd_scores, 'micro-RNA': cadd_mir, 'Domain':cadd_domains, 'Variation Type':cadd_mutation_types, 'Variation location':cadd_locations, 'Motifs': cadd_motif, 'Chromatin State':cadd_cHmm + cadd_encode, 'Splicing':cadd_splice, 'GC Content':cadd_gc, 'TFs':cadd_TF}
trace_features_gruops = ['Absolute Expression', 'Preferential Expression', 'Process Activity', 'Differential PPIs', 'PPIs', 'Paralog Relationships', 'eQTL', 'Expression Variability','Expression during Development', 'Expression variability during Development']
cadd_features_gruops = [c for c in feature_gruops_dict.keys() if c not in trace_features_gruops]
print('cadd_features_gruops', cadd_features_gruops)
feature_gruops_dict2 = {}

for key in feature_gruops_dict:
    for f in feature_gruops_dict[key]:
        feature_gruops_dict2[f] = key
Feature_Groups_df = pd.DataFrame.from_dict(feature_gruops_dict2, orient='index')
Feature_Groups_df.reset_index(inplace=True)
Feature_Groups_df.rename(columns={'index':'col_name', 0:'Features_Group'}, inplace=True)
print(Feature_Groups_df)
path = os.path.join('../..', '..', 'Final_Results', 'Random_Forest', 'Shap_Importance_New', 'Group_Normalized_Importance', 'Features_Group_Clasification.csv')

" ---------------------- Calculate the global importance for each of the feature groups -------------------"

importance_directory = os.listdir(os.path.join('../..', '..', 'Final_Results', 'Random_Forest', 'Shap_Importance_New', 'Values_hg37'))
print(importance_directory)
all_importance_df_list = []

palette ={"Genetic": "cornflowerblue", "General Tissue": "grey", "Tissue-Specific": "indianred"}

for file in importance_directory:
    path = os.path.join('../..', '..', 'Final_Results', 'Random_Forest', 'Shap_Importance_New', 'Values_hg37', file)
    tissue = file.split('_')[0]
    print(tissue)
    Importance = pd.read_csv(path)
    Importance = pd.merge(Importance, Feature_Groups_df, on='col_name')
    Importance['Tissue'] = tissue
    Importance['Kind'] = 'Genetic'
    tissue_word = tissue.split(' ')[0]
    if 'brain' in tissue:
        tissue_word = 'Brain'
    elif 'Whole' in tissue:
        tissue_word = 'Blood'
    elif 'kidney' in tissue:
        tissue_word = 'Kidney'
    Importance.loc[Importance.Features_Group.isin(trace_features_gruops), 'Kind'] = 'General Tissue'
    Importance.loc[Importance['col_name'].str.contains('|'.join([tissue_word, tissue_word.lower()])), 'Kind'] = 'Tissue-Specific'
    sum_importance = Importance['feature_importance_vals'].sum()
    Importance['Normalized Importance'] = Importance['feature_importance_vals'].div(sum_importance)
    print(Importance)
    all_importance_df_list.append(Importance)

" ----------------- Merge all the tissues normalized importance data into one dataframe ----------"
All_Tissues_Importance = pd.concat(all_importance_df_list, axis=0)
print(All_Tissues_Importance)

Sum_Gruop_Importance = All_Tissues_Importance[All_Tissues_Importance['Kind']!='General Tissue'].groupby('Features_Group')['Normalized Importance'].mean()#median
Sum_Gruop_Importance = pd.DataFrame(Sum_Gruop_Importance)
Sum_Gruop_Importance.sort_values('Normalized Importance', inplace=True, ascending=False)
print(Sum_Gruop_Importance)

ordered_groups = Sum_Gruop_Importance.index.tolist()
print(ordered_groups)


CADD_Features = All_Tissues_Importance[All_Tissues_Importance['Kind']=='Genetic']
print(CADD_Features)
cadd_groups = CADD_Features['Features_Group'].unique().tolist()
cadd_order = [c for c in ordered_groups if c in cadd_groups]
TRACE_Features = All_Tissues_Importance[All_Tissues_Importance['Kind']!='Genetic']
print(TRACE_Features)
trace_groups = TRACE_Features['Features_Group'].unique().tolist()
trace_order = [c for c in ordered_groups if c in trace_groups]

" ------------------ Compare between concept groups and Plot the overall Importance -----------"

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(35, 25))
sns.boxplot(y="Features_Group", x="Normalized Importance", data=All_Tissues_Importance, hue='Kind', showfliers=False, order=ordered_groups, palette=palette, ax=axes, width=0.5)#, order=ordered_groups, palette=color_dict, [All_Tissues_Importance['Kind']!='General Tissue']

axes.set_xlim([0, 0.009])
axes.set_ylabel(' ')

test_results_dict = {}
for fg in TRACE_Features['Features_Group'].unique():
    tissue_specific_values = TRACE_Features['Normalized Importance'][(TRACE_Features['Kind'] == 'Tissue-Specific')&(TRACE_Features['Features_Group'] == fg)].tolist()
    other_tissue_values = TRACE_Features['Normalized Importance'][(TRACE_Features['Kind'] == 'General Tissue')&(TRACE_Features['Features_Group'] == fg)].tolist()
    test_results = sts.mannwhitneyu(tissue_specific_values, other_tissue_values, alternative='greater')
    print('@', fg)
    print(test_results)
    test_results_dict[fg] = test_results[1]
print(test_results_dict)
stars_dict = {}
c = 0
for key in test_results_dict:
    if test_results_dict[key] < (2.7182**-10):
        stars_dict[key] = '***'
    elif test_results_dict[key] < (2.7182**-5):
        stars_dict[key] = '**'
    elif test_results_dict[key] < (0.05):
        stars_dict[key] = '*'
    else:
        stars_dict[key] = 'ns'

    plt.text(0.008, ordered_groups.index(key),  stars_dict[key], ha='center', va='bottom', color='k', fontsize=20)
    c +=1
path = os.path.join('../..', '..', 'Final_Results', 'Random_Forest', 'Shap_Importance_New', 'Group_Normalized_Importance','Group_Normalized_Importance_Right.pdf')
plt.savefig(path)
