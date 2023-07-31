import csv
import os


def create_queries_interactinos(filename, output_file):
    """
    This function will create a query to insert all the interactions from the input file
    """
    if(os.path.exists(output_file)):
        os.remove(output_file)
                  
    with open(filename, 'r') as f:
        interactions = list(csv.reader(f, delimiter='\t'))[1:]
    
    for interaction in interactions:
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
            f.write('"INSERT OR IGNORE INTO ppi (s_name_1, s_name_2, ppi_type_id, confidence_score, is_directional, one_is_acting) values (\'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\');",\n'.format(interaction[0].split('.')[1], interaction[1].split('.')[1], ppi_type, interaction[-1], interaction[-3], interaction[-2]))
        

if __name__ == '__main__':
    create_queries_interactinos("Data_retriever\downloaded_data\\511145.protein.actions.v11.0.txt", "Data_retriever\queries_folder\e_coli_interactions_queries.txt")