import sys
import json
import numpy as np
from sequence_similarity import levenshtein_distance, damereu_levenshtein_distance


def find_useful_terms(data: dict or json, protein: str, protein2: str):
    """Find the useful terms for the protein, 
        and after comparing the useful terms with the second protein
        return "positive" or "negative" if the second protein is activated or inhibited.
        data: the data containing the GO terms
        protein: the protein to find the terms for
        protein2: the second protein
        return "positive" or "negative" or "" depending on the result"""
    molecular_function = np.array(data[protein]["molecular function"])
    biological_process = np.array(data[protein]["biological process"])

    for mol_fun in molecular_function:
        if "positive regulation" in mol_fun:
            term = mol_fun.replace("positive ", "")
            result = compare_terms_update(data, term, protein2, 2)
            if(result):
                # print(f"{protein} activates {protein2}, because of {mol_fun}")
                return "positive"
        if "negative regulation" in mol_fun:
            term = mol_fun.replace("negative ", "")
            result = compare_terms_update(data, term, protein2, 2)
            if(result):
                # print(f"{protein} inhibits {protein2}, because of {mol_fun}")
                return "negative"
            
    for bio_proc in biological_process:
        if "positive regulation" in bio_proc:
            term = bio_proc.replace("positive ", "")
            result = compare_terms_update(data, term, protein2, 2)
            if(result):
                # print(f"{protein} activates {protein2}, because of {bio_proc}")
                return "positive"
        if "negative regulation" in bio_proc:
            term = bio_proc.replace("negative ", "")
            result = compare_terms_update(data, term, protein2, 2)
            if(result):
                # print(f"{protein} inhibits {protein2}, because of {bio_proc}")
                return "negative"
    
    return ""
    



def compare_terms_l(data: dict or json, term: str, protein2: str, k: int):
    """Compare the term with the molecular function and biological process of the second protein.
        If the term is similar to one of the terms, return True.
        If not, return False.
        data: the data containing the GO terms
        term: the term to compare
        proteni2: the second protein
        k: the maximum difference between the terms
        return True or False depending on if the term is similar to one of the terms"""
    molecular_function = np.array(data[protein2]["molecular function"])
    biological_process = np.array(data[protein2]["biological process"])
    term_extra = term.replace("regulation of ", "")
    
    for mol_fun in molecular_function:
        score1 = levenshtein_distance(term, mol_fun)
        score2 = levenshtein_distance(term_extra, mol_fun)
        if score1 < k or score2 < k:
            # print(score1, score2)
            # print(f"{term} is similar to {mol_fun}")
            # if(protein2 == "CDC42"):
            #     return "Special case"
            return True
        
    for bio_proc in biological_process:
        score1 = levenshtein_distance(term, bio_proc)
        score2 = levenshtein_distance(term_extra, bio_proc)
        if score1 < k or score2 < k:
            # print(score1, score2)
            # print(f"{term} is similar to {bio_proc}")
            return True
        
    return False

def compare_terms_update(data: dict or json, term: str, protein2: str, k: int):
    """Compare the term with the molecular function and biological process of the second protein.
        If the term is similar to one of the terms, return True.
        If not, return False.
        data: the data containing the GO terms
        term: the term to compare
        proteni2: the second protein
        k: the maximum difference between the terms
        return True or False depending on if the term is similar to one of the terms"""
    molecular_function = np.array(data[protein2]["molecular function"])
    biological_process = np.array(data[protein2]["biological process"])
    term_extra = term.replace("regulation of ", "")
    
    for mol_fun in molecular_function:
        if(term_extra in mol_fun):
            return True
        
    for bio_proc in biological_process:

        if term_extra in bio_proc:
            return True
        
    return False



# def compare_terms_d(data: dict or json, term: str, protein2: str, k: int):
#     """Compare the term with the molecular function and biological process of the second protein.
#         If the term is similar to one of the terms, return True.
#         If not, return False.
#         data: the data containing the GO terms
#         term: the term to compare
#         proteni2: the second protein
#         k: the maximum difference between the terms"""
#     molecular_function = np.array(data[protein2]["molecular function"])
#     biological_process = np.array(data[protein2]["biological process"])
#     term_extra = term.replace("regulation of ", "")
    
#     for mol_fun in molecular_function:
#         if damereu_levenshtein_distance(term, mol_fun) < k or damereu_levenshtein_distance(term_extra, mol_fun) < k:
#             return True
#     for bio_proc in biological_process:
#         if damereu_levenshtein_distance(term, bio_proc) < k or damereu_levenshtein_distance(term_extra, bio_proc) < k:
#             return True
#     return False


