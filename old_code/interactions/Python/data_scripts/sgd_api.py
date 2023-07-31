import requests
import json
from data_creation import create_json_file


def create_json_files(filename: str, output_filename_go: str, output_filename_interaction: str):
    """
    This function creates two json files, one for the GO terms and one for the interactions.
    filename: the name of the file containing the genes
    output_filename_go: the name of the output file for the GO terms
    output_filename_interaction: the name of the output file for the interactions
    """
    genes_info_go, genes_info_interaction = read_file(filename)
    
    create_json_file(genes_info_go, output_filename_go)
    create_json_file(genes_info_interaction, output_filename_interaction)
   
        
def read_file(filename: str):
    """
    This function reads the file containing the genes and returns a dictionary with the GO terms and a dictionary with the interactions.
    filename: the name of the file containing the genes
    return a dictionary with the GO terms and a dictionary with the interactions
    """
    genes_info_go = {}
    genes_info_interaction = {}
    
    with open(filename) as f:
        lines = f.read().splitlines()
        
    for gene in lines:
        print(gene)
        go = go_info(gene)
        interaction = interaction_info(gene)
        genes_info_go.update({gene: go})
        genes_info_interaction.update({gene: interaction})
    
    f.close()
    return genes_info_go, genes_info_interaction 
    
         
def connect_to_api(gene: str, addition: str):
    """
    This function connects to the SGD API and returns the data.
    gene: the name of the gene
    addition: the addition to the url
    return the data in a json format
    """
    SGD_BASE_URL = 'https://www.yeastgenome.org/backend/locus/'
    url = SGD_BASE_URL + gene + addition
    try:
        response = requests.get(url=url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(f"Gene: {gene} not found in the SGD database.")
        raise SystemExit(err)
    
    return response.json()

        
        
def go_info(gene: str):
    """
    This function returns a dictionary with the GO terms.
    gene: the name of the gene
    return a dictionary with the GO terms
    """
    print(gene)
    go_term = '/go_details'
    data = connect_to_api(gene, go_term)
    N = len(data)
    
    molecular_function = []
    biological_process = []
    cellular_component = []
    
    for c, info in enumerate(data):
        if(info['go']['go_aspect'] == 'molecular function' and info['go']['display_name'] not in molecular_function):
            molecular_function.append(info['go']['display_name'])
        elif(info['go']['go_aspect'] == 'biological process' and info['go']['display_name'] not in biological_process):
            biological_process.append(info['go']['display_name'])
        elif(info['go']['go_aspect'] == 'cellular component' and info['go']['display_name'] not in cellular_component):
            cellular_component.append(info['go']['display_name'])

    return {'molecular function': molecular_function, 
            'biological process': biological_process,
            'cellular component': cellular_component}
    
    
def interaction_info(gene: str):
    """
    This function returns a dictionary with the interactions.
    gene: the name of the gene
    return a dictionary with the interactions
    """
    interaction_term = '/interaction_details'
    data = connect_to_api(gene, interaction_term)
    N = len(data)
    
    interactions = []
    
    for c, interaction in enumerate(data):
        if interaction['locus2']['display_name'] != gene:
            connection = interaction['locus2']['display_name']
        else:
            connection = interaction['locus1']['display_name']
            
        if(interaction['interaction_type'] == 'Physical' and connection not in interactions):
            interactions.append([connection, interaction['experiment']['display_name']])
    
    return interactions
   
    
if __name__ == '__main__':    
    create_json_files('Code\Data\werner_proteins.txt', 'Code\Data\genes_info_go.json', 'Code\Data\genes_info_interaction.json')

