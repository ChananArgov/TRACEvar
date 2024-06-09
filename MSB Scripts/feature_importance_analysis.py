"""
Feature importance analysis statistics code
"""

import pandas as pd
import os
from scipy.stats import mannwhitneyu, wilcoxon
import statistics as st


def statistics_test():

    ### Read feature importance values per model
    arr = os.listdir(path)

    ### MannWhitney statistics results dictionary
    statistics_results = {'Model': [], 'Rank_p_value_affected_tissue_vs_variant_less': [],
                          'Rank_p_value_affected_tissue_vs_unaffected_tissue_less': [],
                          'Rank_p_value_variant_vs_unaffected_tissue_less': [],
                          }

    for file in arr:
        if file not in ['Transfer_learning_hg37_SHAP_Importance_Slim.csv', 'All_Tissue_Shap_Importance_Slim_without_subBrains.csv']:

            file_read = pd.read_csv(path + file)
            statistics_results['Model'].append(file.replace('_Shap_Importance_Slim.csv', ''))

            ### lists that hold the ranking of the features
            variant = []
            d_tissue = []
            und_tissue = []

            for rowNum, row in file_read.iterrows():
                if row['Kind'] == 'Variant-Specific':
                    variant.append(row['Rank'])
                elif row['Kind'] == 'Tissue-specific, affected tissue':
                    d_tissue.append(row['Rank'])
                else:
                    und_tissue.append(row['Rank'])

            u, p = mannwhitneyu(d_tissue, variant, alternative='less')
            statistics_results['Rank_p_value_affected_tissue_vs_variant_less'].append(p)
            u, p = mannwhitneyu(d_tissue, und_tissue, alternative='less')
            statistics_results['Rank_p_value_affected_tissue_vs_unaffected_tissue_less'].append(p)
            u, p = mannwhitneyu(variant, und_tissue, alternative='less')
            statistics_results['Rank_p_value_variant_vs_unaffected_tissue_less'].append(p)

    pd.DataFrame.from_dict(statistics_results).to_csv('Rank_statistic_results.csv', index=False)


def statistic_test_transfer_learning_model():
    data = pd.read_csv(path + 'Transfer_learning_hg37_SHAP_Importance_Slim.csv')
    tissue_rank = []
    variant_rank = []

    for rowNum, row in data.iterrows():
        if row['Kind'] == 'Tissue-Specific':
            tissue_rank.append(row['Rank'])
        else:
            variant_rank.append(row['Rank'])
    print('Transfer learning statistics results')
    print('Tissue-specific vs variant-specific',mannwhitneyu(tissue_rank, variant_rank, alternative='less'))


def all_tissue_model_paired_test():
    data = pd.read_csv(path + 'All_Tissue_Shap_Importance_Slim_without_subBrains.csv')



    ### lists hold the median rankings per tissue
    dis_tissue_median = []
    unafect_tissue_median = []
    variant_median = []

    ### lists that holds the ranking of a certain tissue
    curr_tissue = ''
    disease_arr = []
    unaffected_arr = []
    variant_arr = []

    for rowNum, row in data.iterrows():

        if row['Tissue'] != curr_tissue:


            if curr_tissue != '':
                dis_tissue_median.append(st.median(disease_arr))
                unafect_tissue_median.append(st.median(unaffected_arr))
                variant_median.append(st.median(variant_arr))

            curr_tissue = row['Tissue']
            disease_arr = []
            unaffected_arr = []
            variant_arr = []

        ### add ranking based on feature's type
        if row['Kind'] == 'Variant-Specific':
            variant_arr.append(row['Rank'])
        elif row['Kind'] == 'Tissue-specific, affected tissue':
            disease_arr.append(row['Rank'])
        else:
            unaffected_arr.append(row['Rank'])


    dis_tissue_median.append(st.median(disease_arr))
    unafect_tissue_median.append(st.median(unaffected_arr))
    variant_median.append(st.median(variant_arr))

    ### Print statistics results
    print('All tissue model statistics results:')
    print('Disease-specific affected tissue vs. variant-specific',wilcoxon(dis_tissue_median, variant_median, alternative='less'))
    print('Disease-specific affected tissue vs. Disease-specific unaffected tissue',wilcoxon(dis_tissue_median, unafect_tissue_median, alternative='less'))
    print('Variant-specific vs. Disease-specific unaffected tissue',wilcoxon(variant_median, unafect_tissue_median, alternative='less'))


path = 'SHAP-new-fixed/Data/'

### MannWhitney statistic test of feature type ranking per each tissue model
statistics_test()

### Paired wilcoxon statistics test of median feature type ranking of tissue
all_tissue_model_paired_test()

### MannWhitney statistic test of feature type ranking for the transfer learning model
statistic_test_transfer_learning_model()