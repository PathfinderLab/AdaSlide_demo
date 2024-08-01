import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str)
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')

args = parser.parse_args()

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_101_TCGA_RCC':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/RCC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"kich":0, "kirc":1, "kirp":2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_102_TCGA_RCC_x4':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/RCC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"kich":0, "kirc":1, "kirp":2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])
    
elif args.task == 'task_103_TCGA_RCC_lambda_010':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/RCC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"kich":0, "kirc":1, "kirp":2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_104_TCGA_RCC_lambda_025':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/RCC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"kich":0, "kirc":1, "kirp":2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_105_TCGA_RCC_lambda_050':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/RCC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"kich":0, "kirc":1, "kirp":2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])
    
elif args.task == 'task_106_TCGA_RCC_lambda_075':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/RCC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"kich":0, "kirc":1, "kirp":2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])
    
elif args.task == 'task_107_TCGA_RCC_lambda_100':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/RCC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"kich":0, "kirc":1, "kirp":2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_108_TCGA_RCC_VAE':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/RCC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict={"kich":0, "kirc":1, "kirp":2},
                            patient_strat=True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_109_TCGA_RCC_VQVAE':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/RCC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict={"kich":0, "kirc":1, "kirp":2},
                            patient_strat=True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_121_TCGA_RCC_VQVAE_lambda_010':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/RCC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict={"kich":0, "kirc":1, "kirp":2},
                            patient_strat=True,
                            patient_voting='maj',
                            ignore=[])
    
elif args.task == 'task_122_TCGA_RCC_VQVAE_lambda_025':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/RCC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict={"kich":0, "kirc":1, "kirp":2},
                            patient_strat=True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_123_TCGA_RCC_VQVAE_lambda_050':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/RCC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict={"kich":0, "kirc":1, "kirp":2},
                            patient_strat=True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_124_TCGA_RCC_VQVAE_lambda_075':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/RCC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict={"kich":0, "kirc":1, "kirp":2},
                            patient_strat=True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_125_TCGA_RCC_VQVAE_lambda_100':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/RCC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict={"kich":0, "kirc":1, "kirp":2},
                            patient_strat=True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_111_TCGA_NSCLC':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/NSCLC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"luad":0, "lusc":1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_112_TCGA_NSCLC_x4':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/NSCLC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"luad":0, "lusc":1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_113_TCGA_NSCLC_lambda_010':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/NSCLC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"luad":0, "lusc":1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_114_TCGA_NSCLC_lambda_025':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/NSCLC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"luad":0, "lusc":1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_115_TCGA_NSCLC_lambda_050':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/NSCLC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"luad":0, "lusc":1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_116_TCGA_NSCLC_lambda_075':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/NSCLC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"luad":0, "lusc":1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_117_TCGA_NSCLC_lambda_100':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/NSCLC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"luad":0, "lusc":1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_118_TCGA_NSCLC_VAE':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/NSCLC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"luad":0, "lusc":1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])
    
elif args.task == 'task_119_TCGA_NSCLC_VQVAE':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/NSCLC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"luad":0, "lusc":1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])
    
elif args.task == 'task_131_TCGA_NSCLC_VQVAE_lambda_010':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/NSCLC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"luad":0, "lusc":1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_132_TCGA_NSCLC_VQVAE_lambda_025':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/NSCLC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"luad":0, "lusc":1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_133_TCGA_NSCLC_VQVAE_lambda_050':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/NSCLC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"luad":0, "lusc":1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_134_TCGA_NSCLC_VQVAE_lambda_075':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/NSCLC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"luad":0, "lusc":1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_135_TCGA_NSCLC_VQVAE_lambda_100':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path='dataset_csv/NSCLC.csv',
                            shuffle=False, 
                            seed=args.seed, 
                            print_info=True,
                            label_dict = {"luad":0, "lusc":1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



