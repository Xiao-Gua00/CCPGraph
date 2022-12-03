from model import *
import fit

from Mol2Graph import Mol2Graph
from rdkit.Chem import AllChem as Chem
from multiprocessing import Pool
import numpy as np

import torch
import random
from torch_geometric.data import Data #, DataLoader
from torch_geometric.loader import DataLoader
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_table_dir', type=str, help='the table of coformer pairs of train set')
parser.add_argument('--test_table_dir', type=str, help='the table of coformer pairs of test set')
parser.add_argument('--mol_dir', type=str, help='path of molecular file')
#parser.add_argument('mean', type=float, help='the mean value of training density')
#parser.add_argument('std', type=float, help='the standard deviation of training density')
parser.add_argument('--model_name', type=str, help='the path of model relsults')
args = parser.parse_args()
print(args)

if __name__ == '__main__':
    fr_train=args.train_table_dir
    fr_test=args.test_table_dir
    molblock=args.mol_dir
    model_name=args.model_name
    #print('step 1')
    
    ############## data pipline ##############
    fr_train = eval(open(fr_train).read())
    fr_test = eval(open(fr_test).read())

    class DataFeat(object):
        def __init__(self, **kwargs):
            for k in kwargs:
                self.__dict__[k] = kwargs[k]

    mol_block = eval(open(molblock).read())
    def make_data(items):
        try:
            mol1 = Chem.MolFromMolBlock(mol_block[items[0]])
            mol2 = Chem.MolFromMolBlock(mol_block[items[1]])
            ############## if the SMILES file is used ##############
            #mol1 = Chem.MolFromSmiles(mol_block[items[0]])
            #mol1 = Chem.MolFromSmiles(mol_block[items[1]])
            
            g1 = Mol2Graph(mol1)
            g2 = Mol2Graph(mol2)
            x = np.concatenate([g1.x, g2.x], axis=0)
            edge_feats = np.concatenate([g1.edge_feats, g2.edge_feats], axis=0)
            e_idx2 = g2.edge_idx+g1.node_num
            edge_index = np.concatenate([g1.edge_idx, e_idx2], axis=0)
            return DataFeat(x=x, edge_feats=edge_feats, edge_index=edge_index, y=np.array([float(items[2])]))
        except: 
            print('Bad input sample:'+items[-1]+'skipped.')
    #print('step 2')

    ############## multiprocessing ##############
    pool = Pool(processes=30)                 
    data_pool_Train = pool.map(make_data, [i for i in fr_train if Chem.MolFromMolBlock(mol_block[i[0]]) != None and Chem.MolFromMolBlock(mol_block[i[1]]) != None]) 
    data_pool_test = pool.map(make_data, [i for i in fr_test if Chem.MolFromMolBlock(mol_block[i[0]]) != None and Chem.MolFromMolBlock(mol_block[i[1]]) != None])
    pool.close()
    pool.join()
    
    #print('step 3')
    ############## make graph data ##############
    Y_Train = np.array([i.y for i in data_pool_Train])
    std = Y_Train.std()
    mean = Y_Train.mean()
    
    loader_Train = []
    for d in data_pool_Train:
        i = Data(x=torch.tensor(d.x),  
             edge_index=torch.tensor(d.edge_index.T, dtype=torch.long),
             edge_attr=torch.tensor(d.edge_feats),
             y=torch.tensor((d.y-mean)/std, dtype=torch.float32))
        loader_Train.append(i)

    loader_test = []
    for d in data_pool_test:
        i = Data(x=torch.tensor(d.x),  
             edge_index=torch.tensor(d.edge_index.T, dtype=torch.long),
             edge_attr=torch.tensor(d.edge_feats),
             y=torch.tensor((d.y-mean)/std, dtype=torch.float32))
        loader_test.append(i)
    #print('step 4')
    ############## loader data ##############
    random.seed(1000)
    random.shuffle(loader_Train)
    train_loader = DataLoader(loader_Train, batch_size=128, shuffle=0, drop_last=True)
    test_loader = DataLoader(loader_test, batch_size=128, shuffle=0)
    #print('step 5')
    ############## fit ##############
    data_loaders = {'train':train_loader, 'valid':test_loader}
    fit.training(CCPGraph, data_loaders, n_epoch=100, save_att=True, snapshot_path='./snapshot_Bayes_Opt/{}//'.format(model_name), mean=mean, std=std)
    #print('step 6')
    