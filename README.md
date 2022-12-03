# CCPGraph
![image](https://github.com/Xiao-Gua00/CCPGraph/blob/main/img/scheme.png)

# About CCPGraph
Cocrystal engineering as an effective way to modify the solid-state properties has inspired great interests from diverse material fields. Crystal density is intimately correlated with important physical properties of the cocrystals, thus it is highly desired to develop a general and accurate model for quickly predicting the cocrystal density. Motivated by the issue and some limitations of previous method, we develop a high accurate and generalized cocrystal density prediction model, based on graph neural network embedded with attention mechanism, and discuss effects of data quality and feature representation on the model. Our model showcases high prediction performance for unseen samples, the R2, MAE and RMSE are 0.9853, 0.0349 g/cm3 and 0.0427 g/cm3. Our model provides a general and simple cocrystal density prediction tool for the related experimental investigations.

# Usage
We implement our graph neural network model using the PyTorch 1.9.0 deep learning framework, and training the models on Nvidia RTX 2080ti GPU and Nvidia RTX 3080ti.

Requirements:
~~~
* Python 3.7.12
* torch 1.19.0+cu102
* Tensorflow-gpu 1.13.1
* RDKit
* Scikit-learn
* Pandas
~~~

Note 1: Our model provides the density value for cocrystal which stoichiometric ratio is 1:1.
Note 2: You need to prepare the input files of each coformer, whose format can be sturcture file, such as 'sdf', 'mol' or sequence file such as SMILES.

For example, you can use CCPGraph for prediction cocrystal density by the following command:
~~~
python main.py --train_table_dir='./data/train_items.table' --test_table_dir='./data/test_items.table' --mol_dir='./data/molblocks.dir' --model_name='save_path'
~~~

# Notification of commercial use
Commercialization of this product is prohibited without our permission. 