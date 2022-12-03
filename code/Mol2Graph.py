import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

import os

att_dtype = np.float32

PeriodicTable = Chem.GetPeriodicTable()
try:
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
except:
    fdefName = os.path.join('/RDKit file path**/RDKit/Data/','BaseFeatures.fdef')  #The 'RDKit file path**' is the installation path of RDKit.
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
possible_atom_type = ['H','B','C','N','O','F','Si','P','S','Cl','Br','I']   
possible_hybridization = ['S','SP','SP2', 'SP3', 'SP3D','SP3D2', 'UNSPECIFIED']
possible_bond_type = ['SINGLE','DOUBLE','TRIPLE','AROMATIC']

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        print(x,type(x))
        raise Exception("input {0} not in allowable set{1}:".format(
                x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def donor_acceptor(rd_mol):
    is_donor = defaultdict(int)
    is_acceptor = defaultdict(int)
    feats = factory.GetFeaturesForMol(rd_mol)
    for i in range(len(feats)):
        if feats[i].GetFamily() == 'Donor':
            for u in feats[i].GetAtomIds():
                is_donor[u] = 1
        elif feats[i].GetFamily() == 'Acceptor':
            for u in feats[i].GetAtomIds():
                is_acceptor[u] = 1
    return is_donor, is_acceptor

def AtomAttributes(rd_atom, is_donor, is_acceptor, extra_attributes=[]):
    
    rd_idx = rd_atom.GetIdx()
    #Inititalize
    attributes = []
    #Add atimic number
    attributes += one_of_k_encoding(rd_atom.GetSymbol(), possible_atom_type)
    #Add heavy neighbor count
    attributes += one_of_k_encoding(len(rd_atom.GetNeighbors()), [0, 1, 2, 3, 4, 5, 6])
    #Add neighbor hydrogen count
    attributes += one_of_k_encoding(rd_atom.GetTotalNumHs(includeNeighbors=True), [0, 1, 2, 3, 4])
    #Add hybridization type
    attributes += one_of_k_encoding(rd_atom.GetHybridization().__str__(), possible_hybridization)
    #Add boolean if chiral
    attributes += one_of_k_encoding(int(rd_atom.GetChiralTag()), [0, 1, 2, 3])
    # Add boolean if in ring
    attributes.append(rd_atom.IsInRing())
    # Add boolean if aromatic atom
    attributes.append(rd_atom.GetIsAromatic())
    #Add boolean if donor
    attributes.append(is_donor[rd_idx])
    #Add boolean if acceptor
    attributes.append(is_acceptor[rd_idx])
    
    attributes += extra_attributes
    return np.array(attributes, dtype=att_dtype)


def atom_featurizer(rd_mol):
    
    is_donor, is_acceptor = donor_acceptor(rd_mol)
    
    #### add atoms descriptors####
    V = []
    for k, atom in enumerate(rd_mol.GetAtoms()):
        all_atom_attr = AtomAttributes(atom, is_donor, is_acceptor) 
        V.append(all_atom_attr)
    return np.array(V, dtype=att_dtype)
    
def bond_featurizer(mol):
    #conf = mol.GetConformer()
    bond_idx, bond_feats = [], []
    for b in mol.GetBonds():
        start = b.GetBeginAtomIdx()
        end = b.GetEndAtomIdx()
        b_type = one_of_k_encoding(b.GetBondType().__str__(), possible_bond_type)
        #start_coor = [i for i in conf.GetAtomPosition(start)]
        #end_coor = [i for i in conf.GetAtomPosition(end)]
        #b_length = np.linalg.norm(np.array(end_coor)-np.array(start_coor))
        #b_type.insert(0, b_length)
        b_type.insert(0, b.GetIsConjugated())  
        b_type.insert(0, b.IsInRing())        
        bond_idx.append([start, end])
        bond_idx.append([end, start])
        bond_feats.append(b_type)
        bond_feats.append(b_type)
    e_sorted_idx = sorted(range(len(bond_idx)), key=lambda k:bond_idx[k])
    bond_idx = np.array(bond_idx)[e_sorted_idx]
    bond_feats = np.array(bond_feats, dtype=np.float32)[e_sorted_idx]
    return bond_idx.astype(np.int64), bond_feats.astype(np.float32) #bond_idx.astype(np.int64).T

class Mol2Graph(object):
    def __init__(self, mol, **kwargs):
        self.x = atom_featurizer(mol)
        self.edge_idx, self.edge_feats = bond_featurizer(mol)
        self.node_num = self.x.shape[0]
        #self.tag = mol.GetProp('_Name')
        for k in kwargs:
            self.__dict__[k] = kwargs[k]