"""
-----------------------------------------------------------------------------
AUTHOR: Soumitra Samanta (soumitramath39@gmail.com)
-----------------------------------------------------------------------------
"""

import subprocess
import os
import numpy as np
from datetime import datetime
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

__all__ = [
    'FragGenie'
]

class FragGenie():
    
    def __init__(self, dir_fraggenie=''):
        
        self.dir_fraggenie = dir_fraggenie
        
    def to_numpy(self, array_str, sep=','):

        return np.fromstring(array_str[1:-1], sep=sep)

    def create_folder(self, folder_name):
        if len(folder_name):
            if not os.path.isdir(folder_name):
                os.makedirs(folder_name)

        return folder_name

    def mol_prop_mass(self, smiles):
        """
        Molecular mass
        """

        return [Descriptors.ExactMolWt(Chem.MolFromSmiles(sm)) for sm in smiles]
    
    def smiles2fraggenie_csv(
        self, 
        input_path='', 
        input_filename='test_input.csv', 
        smiles_col='smiles',
        output_path='', 
        output_filename='',
        num_bonds_to_break=3, 
        min_fragment_mass=50,
        max_smiles_len=250, 
        max_num_smiles=1000000000, 
        flag_display='true',
        masses_option='METFRAG_MZ'
    ):
        """Calculate FragGenie from csv file"""

        if(len(output_path)==0):
            output_path = input_path
        if(len(output_filename)==0):
            output_filename = ''.join([
                'fraggenie_', datetime.today().strftime('%d%m%Y%H%M%S'), 
                '_', str(np.random.random(1)[0])[2:], 
                '_nbonds_', str(num_bonds_to_break), 
                '_frgms_', str(min_fragment_mass), 
                '_smlen_', str(max_smiles_len),
                '_', input_filename
            ])
        bash_cmd = ''.join([
            'bash ', self.dir_fraggenie, 
            'fragment.sh ', 
            input_path, 
            input_filename, 
            ' ', output_path, 
            output_filename, 
            ' ', smiles_col, 
            ' ', str(num_bonds_to_break), 
            ' ', str(min_fragment_mass), 
            ' ', str(max_smiles_len), 
            ' ', str(max_num_smiles), 
            ' ', flag_display, 
            ' ', masses_option
        ])

        subprocess.call(bash_cmd, shell=True)

        return output_path, output_filename, bash_cmd

    def smiles2fraggenie(
        self, 
        smiles,  
        num_bonds_to_break=3, 
        min_fragment_mass=50, 
        max_smiles_len=250, 
        max_num_smiles=1000000000, 
        flag_display='true',
        masses_option='METFRAG_MZ',
        input_path='dump/', 
        input_filename='', 
        massspec_sep=',', 
        fill_non_break_mol=1, 
        flag_del_temp_file=1,
        verbose=0
    ):
        """Calculate FragGenie from smiles"""

        input_path = self.create_folder(input_path)
        if len(input_filename)==0:
            input_filename = ''.join(['smiles_', datetime.today().strftime('%d%m%Y%H%M%S'), 
                                      '_', str(np.random.random(1)[0])[2:], 
                                      '.csv'
                                     ])

        pd.DataFrame.from_dict({'smiles':smiles}).to_csv(''.join([input_path, input_filename]), index=False)

        output_path, output_filename, bash_cmd = self.smiles2fraggenie_csv(
            input_path=input_path, 
            input_filename=input_filename, 
            num_bonds_to_break=num_bonds_to_break, 
            min_fragment_mass=min_fragment_mass, 
            max_smiles_len=max_smiles_len,
            max_num_smiles=max_num_smiles,
            flag_display=flag_display, 
            masses_option=masses_option
        )


        df_smiles = pd.read_csv(output_path+output_filename)

        # handle very small molecules which is unable to break into fraggenie (fill with mol mass) or unbreakable molecules
        if fill_non_break_mol:
            fraggenie = [None]*len(smiles)
            fraggenie_smiles = df_smiles['smiles'].tolist()
            count1 = 0
            count2 = 0
            for i, sm in enumerate(smiles):
                try:
                    fraggenie[i] = self.to_numpy(df_smiles[masses_option][fraggenie_smiles.index(sm)], sep=massspec_sep)
                    if len(fraggenie[i])==0:
                        if verbose:
                            print('Unable to break molecules: {}-{}' .format(i, smiles[i]))
                        fraggenie[i] = np.asarray([self.mol_prop_mass([smiles[i]])[0]])
                        count1 += 1
                except:
                    if verbose:
                        print('Unable to break molecules: {}-{}' .format(i, smiles[i]))
                    fraggenie[i] = np.asarray([self.mol_prop_mass([smiles[i]])[0]])
                    count2 += 1
            print('Total number of unbreakable molecules: {} (empty-{}, not all-{})' .format(count1+count2, count1, count2))
        else:
            fraggenie = df_smiles[masses_option].apply(self.to_numpy, sep=massspec_sep).tolist()
            

        if flag_del_temp_file:
            filename = ''.join([input_path, input_filename])
            if os.path.isfile(filename):
                if verbose:
                    print('Removing "{}"' .format(filename))
                os.remove(filename)
            filename = ''.join([output_path, output_filename])
            if os.path.isfile(filename):
                if verbose:
                    print('Removing "{}"' .format(filename))
                os.remove(filename)


        return fraggenie
    
if __name__ == '__main__':
    
    fraggenie = FragGenie()
    
    output_path, output_filename, bash_cmd =  fraggenie.smiles2fraggenie_csv(output_filename='fraggenie_test_input.csv')
    
    
    smiles = ['Cn1cnc2n(C)c(=O)n(C)c(=O)c12', 
              'BrC1CCCCc1CC', 
              'C#1C#CC1', 
              'C#1C#CCcCCCc1', 
              'C#1CCCCCCC=1', 
              'C#1CCcNccccccccc1', 
              'Cn1cnc2n(C)c(=O)n(C)c(=O)c12']
    
    fragment = fraggenie.smiles2fraggenie(smiles, fill_non_break_mol=1)
    
    for i in range(len(smiles)):
        print('smiles: {}\nfragment: {}' .format(smiles[i], fragment[i]))
    
    