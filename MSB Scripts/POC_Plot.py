"""
Plotting POC examples, Fig 2A-B.
"""



import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

root_directory = 'Your relevant directory'

"----------------------------------------- Load POC Results And Creat Plots --------------------------------"

def case_example(gene_name, disease_tissue):
    
    path = os.path.join(root_directory, gene_name + '_Predictions_Example.csv')
    Example_data = pd.read_csv(path)
    print(Example_data)
    tissues = list(Example_data)[22:]
    print(tissues)
    pathogenic_probability_list = []
    Example_data['Variant'] = Example_data['Pos'].astype(str)   + '_' +  Example_data['Type']#Example_data['#Chr']
    tissue_dict = {'brain': 'Brain', 'Heart - Left Ventricle': 'Heart', 'kidney': 'Kidney', 'Lung':'Lung', 'Testis': 'Testis', 'Whole Blood': 'Blood'}
    for tissue in tissues:
        tissue_prob = Example_data[tissue].copy()
        tissue_prob = pd.concat([Example_data[['Pos', 'Variant']], tissue_prob], axis=1)

        tissue_prob['Tissue'] = tissue_dict[tissue]
        tissue_prob = tissue_prob.rename(columns={tissue: 'Pathogenic Score'})
        pathogenic_probability_list.append(tissue_prob)



    Pathogenic_Prob = pd.concat(pathogenic_probability_list)
    Pathogenic_Prob = Pathogenic_Prob.sort_values('Variant')
    print(tissues)
    order = ['Blood', 'Brain', 'Heart', 'Kidney','Lung', 'Testis']
    palette = [ 'gray',  'blue', 'red', 'green', 'orange', 'plum', 'lightseagreen']

    sns.catplot( x="Variant", y="Pathogenic Score", hue="Tissue", data=Pathogenic_Prob, kind="strip", height=3, aspect=4, hue_order=order, s=30, palette = 'Dark2')#, palette = 'Set1')#'husl''husl'

    Example_data = Example_data.sort_values('Pos', ignore_index=True)
    print(Example_data)
    patogenic_indexes = Example_data[Example_data[disease_tissue] == True].index
    print(patogenic_indexes)
    for ind in patogenic_indexes:
        print(ind)
        plt.gca().get_xticklabels()[ind].set_color("red")

    plt.subplots_adjust(left=0.1, bottom=0.4, right=0.8, top=0.8, wspace=0, hspace=0)
    plt.xticks(fontsize=5, rotation=90)
    plt.tick_params(labelsize=8)
    plt.suptitle(gene_name)

    path = os.path.join(root_directory, gene_name + '_Predictions_Example.pdf')

    plt.savefig(path)

    plt.show()


gene_name = 'AHDC1'
disease_tissue = 'brain_disease_causing'

case_example(gene_name, disease_tissue)
