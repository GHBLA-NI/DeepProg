import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from simdeep.simdeep_boosting import SimDeepBoosting
from simdeep.config import PATH_THIS_FILE
from sklearn.preprocessing import RobustScaler
from collections import OrderedDict
from os.path import isfile
import numpy as np
import random
import pandas as pd
import os
import scipy.stats as st
import dill
from simdeep.simdeep_utils import save_model
from simdeep.simdeep_utils import load_model
# specify your data path
from argparse import ArgumentParser


parser = ArgumentParser(description="data generate")
parser.add_argument('--selection', '-S',
                    type=str,
                    help="Bandwidth of host links (Mb/s)",
                    default='1')
parser.add_argument('--dataset', '-D',
                    type=str,
                    help="name of the dataset",
                    default='HCC')
args = parser.parse_args()



# Add the parameter strategy selection
selection = args.selection
dataset = args.dataset

path_data = str(Path(__file__).resolve().parent.parent / 'data' / dataset) + '/'
parameter_df = pd.read_csv(str(Path(__file__).resolve().parent.parent / 'data' / 'hyperparameter.csv'), sep=',')

if selection == "0":
    tsv_files = OrderedDict([
     ('MIR', 'mir.tsv'),
     ('METH', 'meth.tsv'),
     ('RNA', 'rna.tsv')
    ])
    PROJECT_NAME = 'single'
    feature_selection_usage = 'individual'
if selection == "1":
    tsv_files = OrderedDict([
     ('MIR', 'mir_cnv.tsv'),
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv')
    ])
    PROJECT_NAME = 'single_testingnode'
    feature_selection_usage = 'individual'
elif selection == "2":
    tsv_files = OrderedDict([
     ('MIR', 'mir_cnv.tsv'),
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'), 
     ("drug","drug.tsv")
    ])  
    PROJECT_NAME = 'drug_single'
    feature_selection_usage = 'individual'
elif selection == "3":
    tsv_files = OrderedDict([
     ('MIR', 'mir_cnv.tsv'),
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'), 
     ("wxs","wxs.tsv")
    ])   
    PROJECT_NAME =  'snp_single'
    feature_selection_usage = 'individual'
elif selection == "4":
    tsv_files = OrderedDict([
     ('MIR', 'mir_cnv.tsv'),
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'), 
     ("percentage", "percentage_scdc.tsv")
    ])   
    PROJECT_NAME =  'dec_single'
    feature_selection_usage = 'individual'
elif selection == "5":
    tsv_files = OrderedDict([
     ('MIR', 'mir_cnv.tsv'),
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'), 
     ("drug","drug.tsv"),
     ("wxs","wxs.tsv")
    ])  
    PROJECT_NAME = 'drug_snp_single'
    feature_selection_usage = 'individual'
elif selection == "6":
    tsv_files = OrderedDict([
     ('MIR', 'mir_cnv.tsv'),
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'), 
     ("percentage", "percentage_scdc.tsv"),
     ("drug","drug.tsv"),
    ])  
    PROJECT_NAME =  'drug_dec_single'
    feature_selection_usage = 'individual'
elif selection == "7":
    tsv_files = OrderedDict([
     ('MIR', 'mir_cnv.tsv'),
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'), 
     ("percentage", "percentage_scdc.tsv"),
     ("drug","drug.tsv"),
    ("wxs","wxs.tsv")
    ])  
    PROJECT_NAME =  'once_drug_snp_dec_single'
    feature_selection_usage = 'individual'
elif selection == "15":
    tsv_files = OrderedDict([
     ('MIR', 'mir_cnv.tsv'),
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'), 
     ("cnv","cnv.tsv")
    ])  
    PROJECT_NAME = 'cnv_single'
    feature_selection_usage = 'individual'
elif selection == "16":
    tsv_files = OrderedDict([
     ('MIR', 'mir_cnv.tsv'),
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'), 
     ("cnv","cnv.tsv"),
     ("drug","drug.tsv")
    ])  
    PROJECT_NAME = 'drug_cnv_single'
    feature_selection_usage = 'individual'
elif selection == "17":
    tsv_files = OrderedDict([
     ('MIR', 'mir.tsv'),
     ('METH', 'meth.tsv'),
     ('RNA', 'rna.tsv'), 
     ("percentage", "percentage_scdc.tsv"),
     ("drug","drug.tsv"),
     ("cnv","cnv.tsv")
    ])  
    PROJECT_NAME =  'drug_dec_cnv_single'
    feature_selection_usage = 'individual'
elif selection == "18":
    tsv_files = OrderedDict([
     ('METH', 'meth.tsv'),
     ('RNA', 'rna.tsv')
    ])  
    PROJECT_NAME =  'single'
    feature_selection_usage = 'individual'
elif selection == "20":
    tsv_files = OrderedDict([
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'),
     ("mir","mir_cnv.tsv"),
     ("clinical","clinical_common_final.tsv")
    ])  
    PROJECT_NAME =  'clinical_single'
    feature_selection_usage = 'individual'
elif selection == "21":
    tsv_files = OrderedDict([
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'),
     ("mir","mir_cnv.tsv"),
     ("clinical","clinical_common_final.tsv")
    ])  
    PROJECT_NAME =  'clinical_common_single'
    feature_selection_usage = 'individual'
elif selection == "22":
    tsv_files = OrderedDict([
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'),
     ("mir","mir_cnv.tsv"),
     ("clinical","clinical_try.tsv"),
     ("percentage", "percentage_scdc.tsv"),
     ("drug","drug.tsv"),
     ("wxs","wxs.tsv")
    ])  
    PROJECT_NAME =  'drug_snp_dec_clinical_single'
    feature_selection_usage = 'individual'
elif selection == "23":
    tsv_files = OrderedDict([
     ("clinical","clinical_common_final.tsv")
    ])  
    PROJECT_NAME =  'clinical_pure'
    feature_selection_usage = 'individual'
elif selection == "24":
    tsv_files = OrderedDict([
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'),
     ("mir","mir_cnv.tsv"),
     ("clinical","clinical_try_all.tsv"),
     ("percentage", "percentage_scdc.tsv"),
     ("drug","drug.tsv"),
     ("wxs","wxs.tsv")
    ])  
    PROJECT_NAME =  'drug_snp_dec_clinical_single_all'
    feature_selection_usage = 'individual'
elif selection == "25":
    tsv_files = OrderedDict([
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'),
     ("mir","mir_cnv.tsv"),
     ("clinical","clinical_common_final.tsv"),
     ("percentage", "percentage_scdc.tsv"),
     ("drug","drug.tsv"),
     ("wxs","wxs.tsv")
    ])  
    PROJECT_NAME =  'drug_snp_dec_clinical_single_common'
    feature_selection_usage = 'individual'
elif selection == "26":
    tsv_files = OrderedDict([
     ('METH', 'meth.tsv'),
     ('RNA', 'rna.tsv'),
     ("mir","mir.tsv"),
     ("clinical","clinical_common_final.tsv"),
     ("percentage", "percentage_scdc.tsv"),
     ("drug","drug.tsv"),
     ("wxs","wxs.tsv")
    ])  
    PROJECT_NAME =  'drug_snp_dec_clinical_single_common'
    feature_selection_usage = 'individual'
elif selection == "27":
    tsv_files = OrderedDict([
     ('METH', 'meth.tsv'),
     ('RNA', 'rna.tsv'),
     ("mir","mir.tsv"),
     ("clinical","clinical_common_final.tsv")
    ])  
    PROJECT_NAME =  'clinical_single'
    feature_selection_usage = 'individual'
elif selection == "28":
    tsv_files = OrderedDict([
     ('MIR', 'mir_cnv.tsv'),
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'), 
     ("cnv","cnv.tsv"),
     ("drug","drug.tsv"),
     ("percentage", "percentage_scdc.tsv")
    ])  
    PROJECT_NAME = 'drug_cnv_dec_single'
    feature_selection_usage = 'individual'
elif selection == "29":
    tsv_files = OrderedDict([
     ('MIR', 'mir_cnv.tsv'),
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'), 
     ("cnv","cnv.tsv"),
     ("drug","drug.tsv"),
     ("wxs","wxs.tsv")
    ])  
    PROJECT_NAME = 'drug_cnv_snp_single'
    feature_selection_usage = 'individual'
elif selection == "30":
    tsv_files = OrderedDict([
     ('MIR', 'mir.tsv'),
     ('METH', 'meth.tsv'),
     ('RNA', 'rna.tsv'), 
     ("drug","drug.tsv"),
    ])  
    PROJECT_NAME = 'drug_single'
    feature_selection_usage = 'individual'
elif selection == "31":
    tsv_files = OrderedDict([
     ('METH', 'meth.tsv'),
     ('RNA', 'rna.tsv'), 
     ("drug","drug.tsv"),
    ])  
    PROJECT_NAME = 'drug_single'
    feature_selection_usage = 'individual'
elif selection == "32":
    tsv_files = OrderedDict([
     ('MIR', 'mir.tsv'),
     ('METH', 'meth.tsv'),
     ('RNA', 'rna.tsv'), 
     ("percentage", "percentage_scdc.tsv"),
     ("drug","drug.tsv"),
    ("wxs","wxs.tsv")
    ])  
    PROJECT_NAME =  'once_drug_snp_dec_single'
    feature_selection_usage = 'individual'
elif selection == "33":
    tsv_files = OrderedDict([
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'), 
     ("percentage", "percentage_scdc.tsv"),
     ("drug","drug.tsv"),
    ("wxs","wxs.tsv")
    ])  
    PROJECT_NAME =  'once_drug_snp_dec_single'
    feature_selection_usage = 'individual'
elif selection == "34":
    tsv_files = OrderedDict([
     ('MIR', 'mir_cnv.tsv'),
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'), 
     ("cnv","cnv.tsv"),
     ("drug","drug_modified.tsv"),
    ("wxs","wxs.tsv")
    ])  
    PROJECT_NAME =  'once_drug_cnv_snp_single_drug_test'
    feature_selection_usage = 'individual'
elif selection == "35":
    tsv_files = OrderedDict([
     ('MIR', 'mir.tsv'),
     ('METH', 'meth.tsv'),
     ('RNA', 'rna.tsv'), 
     ("cnv","cnv.tsv"),
     ("drug","drug_modified.tsv"),
    ("wxs","wxs.tsv")
    ])  
    PROJECT_NAME =  'once_drug_cnv_snp_single_drug_test'
    feature_selection_usage = 'individual'
elif selection == "36":
    tsv_files = OrderedDict([
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'), 
     ("cnv","cnv.tsv"),
     ("drug","drug_modified.tsv"),
    ("wxs","wxs.tsv")
    ])  
    PROJECT_NAME =  'once_drug_cnv_snp_single_drug_test'
    feature_selection_usage = 'individual'
elif selection == "37":
    tsv_files = OrderedDict([
     ('METH', 'meth_cnv.tsv'),
     ('MIR', 'mir_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'), 
     ("cnv","cnv.tsv"),
    ("wxs","wxs.tsv")
    ])  
    PROJECT_NAME =  'once_drug_cnv_snp_single_drug_ANOVA'
    feature_selection_usage = 'individual'
elif selection == "38":
    tsv_files = OrderedDict([
     ('METH', 'meth_cnv.tsv'),
     ('MIR', 'mir_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'), 
     ("drug","drug_modified.tsv"),
     ("cnv","cnv.tsv"),
    ("wxs","wxs.tsv")
    ])  
    PROJECT_NAME =  'once_drug_cnv_snp_single_drug_TAU_cindextest'
    feature_selection_usage = 'individual'
elif selection == "39":
    tsv_files = OrderedDict([
     ('METH', 'meth_cnv.tsv'),
     ('MIR', 'mir_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'), 
     ("drug","drug_modified.tsv")
    ])  
    PROJECT_NAME =  'drug_single_TAU'
    feature_selection_usage = 'individual'
elif selection == "40":
    tsv_files = OrderedDict([
     ('METH', 'meth_cnv.tsv'),
     ('RNA', 'rna_cnv.tsv'), 
     ("drug","drug_modified.tsv")
    ])  
    PROJECT_NAME =  'drug_single_TAU'
    feature_selection_usage = 'individual'
elif selection == "41":
    tsv_files = OrderedDict([
     ('METH', 'meth.tsv'),
     ('MIR', 'mir.tsv'),
     ('RNA', 'rna.tsv'), 
     ("drug","drug_modified.tsv")
    ])  
    PROJECT_NAME =  'drug_single_TAU'
    feature_selection_usage = 'individual'
# The survival file located also in the same folder
survival_tsv = 'survival.tsv'

assert(isfile(path_data + "survival.tsv"))

# More attributes
# Create the seed list randomly so that in the for loop for implementation, the seed is different.
seed_list = []
for i in range(50):
    seed_list.append(random.randint(0,1000))
#     seed_list.append(i*10)
print(seed_list)


data_parameter = parameter_df[parameter_df['cancer_type'] == dataset]


for idx, SEED in enumerate(seed_list): 
    EPOCHS = 10# autoencoder fitting epoch
    nb_it = 10 # Number of submodels to be fitted
    nb_threads = 2 # Number of python threads used to fit survival model
    cluster_method = data_parameter['cluster_method'].values[0]
    class_selection = data_parameter['class_selection'].values[0]
    survival_flag = {
        'patient_id': 'Samples',
        'survival': 'days',
        'event': 'event'}
    
    output_dir = str(Path(__file__).resolve().parent.parent / 'result')
    os.makedirs(output_dir, exist_ok=True)  

    boosting = SimDeepBoosting(
        nb_threads=nb_threads,
        nb_it=nb_it,
        split_n_fold=2,
        survival_tsv=survival_tsv,
        training_tsv=tsv_files,
        path_data=path_data,
        project_name=PROJECT_NAME,
        path_results=output_dir,
        epochs=EPOCHS,
        survival_flag=survival_flag,
        distribute=False,
        seed=SEED,
        nb_clusters = int(data_parameter['nb_clusters'].values[0]),
        use_autoencoders = True,
        use_r_packages= False,
        cluster_method = cluster_method,
        feature_selection_usage = feature_selection_usage,
        class_selection = class_selection
        )
    boosting.fit()
    boosting.predict_labels_on_full_dataset()
