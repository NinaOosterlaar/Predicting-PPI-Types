import os
import numpy as np
import sys
sys.path.append("intearctions/Python/data_scripts")
from check_regulation import find_useful_terms
from sgd_api import go_info
from data_creation import create_json_file
from retrieve_data import retrieve_json_data, retrieve_csv_data


def protein_interactions_and_go_file(interaction_file: str, go_file: str):
    """Loop trough the interactions and retrieve the useful Go terms.
        If a protein does not have GO terms, it will be added trough the SGD API.
        interaction_file: the name of the file containing the interactions
        go_file: the name of the file containing the GO terms
        return a list containing the interactions and the regulation"""
    if(os.path.exists(go_file)):
        data = dict(retrieve_json_data(go_file))
    else: 
        data = {}
    
    result = []
    length = len(data)
        
    ppis = retrieve_csv_data(interaction_file)

    for ppi in ppis:
        if(ppi[0] not in data):
            data[ppi[0]] = go_info(ppi[0])
        if(ppi[1] not in data):
            data[ppi[1]] = go_info(ppi[1])
        
        result_p0 = find_useful_terms(data, ppi[0], ppi[1])
        # result_p1 = find_useful_terms(data, ppi[1], ppi[0])
        
        if(result_p0):
            if(result_p0 == "positive"):
                result.append([ppi[0], ppi[1], "activation"])
            else:
                result.append([ppi[0], ppi[1], "inhibition"])
        # if(result_p1):
        #     if(result_p1 == "positive"):
        #         result.append([ppi[1], ppi[0], "activating"])
        #     else:
        #         result.append([ppi[1], ppi[0], "inhibiting"])
        

    if(len(data) > length):
        create_json_file(data, go_file)
    
    return result
        
        
        
if (__name__ == "__main__"):
    protein_interactions_and_go_file(
        "Code/Data/protein-interactions.csv", "Code/Data/genes_info_go.json")