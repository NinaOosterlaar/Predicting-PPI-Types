import os
import csv


def create_gene(input_file, output_file = None):
    """
    This function will create a query to insert all the genes from the input file
    """
    if output_file is not None:
        if os.path.exists(output_file):
            os.remove(output_file)
        
    with open(input_file, 'r') as f:
        lines = f.readlines()[5:]
    
    genes = set()
    
    for line in lines:
        line = line.split('\t')
        gene = line[1]
        genes.add(gene)
    
    for gene_unique in genes:
        with open(output_file, 'a') as f:
            f.write('"INSERT OR IGNORE INTO gene (s_name, organism_id) values (\'{}\', \'{}\');",\n'.format(gene_unique, 4))
    
    return genes
            

def create_gene_go_query(input_file, output_file):
    """
    This function will create a query to insert all the GO terms from all the genes from the input file
    """
    if os.path.exists(output_file):
        os.remove(output_file)
    
    with open(input_file, 'r') as f:
        lines = f.readlines()[5:]
    
    for line in lines:
        line = line.split('\t')
        gene = line[1]
        go_term = line[4]
        with open(output_file, 'a') as f:
            f.write('"INSERT OR IGNORE INTO gene_go_term (s_name, go_id) values (\'{}\', \'{}\');",\n'.format(gene, go_term))
            

def create_queries_interactinos(filename, output_file):
    """
    This function will create a query to insert all the interactions from the input file
    """
    if(os.path.exists(output_file)):
        os.remove(output_file)
                  
    with open(filename, 'r') as f:
        interactions = list(csv.reader(f, delimiter='\t'))[1:]
        
    genes = frozenset(create_gene("Data_retriever\downloaded_data\gene_association.pombase"))
    
    for interaction in interactions:
        if interaction[0][5:-2] in genes and interaction[1][5:-2] in genes:
            with open(output_file, 'a') as f:
                if(interaction[2] == 'activation'):
                    ppi_type = 1
                elif(interaction[2] == 'inhibition'):
                    ppi_type = 2
                elif(interaction[2] == 'binding'):
                    ppi_type = 3
                elif(interaction[2] == 'catalysis'):
                    ppi_type = 4
                elif(interaction[2] == 'ptmod'):
                    ppi_type = 5
                elif(interaction[2] == 'reaction'):
                    ppi_type = 6
                elif(interaction[2] == 'expression'):
                    ppi_type = 7
                else:
                    ppi_type = 8
                f.write('"INSERT OR IGNORE INTO ppi (s_name_1, s_name_2, ppi_type_id, confidence_score, is_directional, one_is_acting) values (\'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\');",\n'.format(interaction[0][5:-2], interaction[1][5:-2], ppi_type, interaction[-1], interaction[-3], interaction[-2]))
        

if __name__ == '__main__':
    # create_gene("Data_retriever\downloaded_data\gene_association.pombase", "Data_retriever\queries_folder\pombe_gene_queries.txt")
    # create_gene_go_query("Data_retriever\downloaded_data\gene_association.pombase", "Data_retriever\queries_folder\pombe_gene_go_queries.txt")
    create_queries_interactinos("Data_retriever\downloaded_data\\4896.protein.actions.v11.0.txt", "Data_retriever\queries_folder\pombe_interactions_queries.txt")