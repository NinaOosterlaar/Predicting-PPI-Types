import csv
import os
import sqlite3 as sq
from sqlite3 import Error

database_path = r"C:\Users\\ninao\OneDrive - Delft University of Technology\Nanobiology\Bachelor End Project\Bep\Database\database.sqlite"

def create_connection(db_file):
    try:
        return sq.connect(db_file)
    except Error as e:
        print(e)
    
    return None

def run(parameter):
    conn = create_connection(database_path)
    try:
        cur = conn.cursor()
        args_str = ','.join(["('{}','{}')".format(*x) for x in parameter])
        cur.execute("INSERT OR IGNORE INTO gene_go_term (s_name, go_id) VALUES " + args_str)
        conn.commit()
        cur.close()
    except Error as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

def retrieve_ensembl_ids(filename, temp_file):
    list_of_ensembl_ids = []    
    with open(filename, 'r') as f:
        interactions = list(csv.reader(f, delimiter='\t'))[1:]
        i = 0
        for interaction in interactions:
            if(interaction[0][5:] not in list_of_ensembl_ids):
                list_of_ensembl_ids.append(interaction[0][5:])
            if(interaction[1][5:] not in list_of_ensembl_ids):
                list_of_ensembl_ids.append(interaction[1][5:])
            i+=1
            if(i%100000 == 0):
                print(i)
    print(len(list_of_ensembl_ids))
    print(list_of_ensembl_ids[:10])
    with open(temp_file, 'w') as f:
       for id in list_of_ensembl_ids:
           f.write("{}\n".format(id))
           
    
def write_human_gene_queries(filename, output_file):
    with open(filename, 'r') as f:
        ensembl_ids = list(csv.reader(f, delimiter='\n'))
    
    if os.path.exists(output_file):
        os.remove(output_file)
    for ensembl_id in ensembl_ids:
        with open(output_file, 'a') as f:
            f.write('"INSERT INTO gene (s_name, organism_id) values (\'{}\', \'{}\');",\n'.format(ensembl_id[0], 2))    
            
            
def write_human_go_queries(input):
    with open(input, 'r') as f:
        genes = list(csv.reader(f, delimiter=','))[1:]
    i = 0
    query_values = []
    for gene in genes:
        if(i%1000 == 0):
            i+=1
        terms = []
        s_name = gene[0]
        
        for term in gene[4:][0].split('; '):
            terms.append(term)
        for GO_term in terms:
            if GO_term == '':
                continue
            # myfile.write('"INSERT INTO gene_go_term (s_name, go_id) values (\'{}\', \'{}\');",\n'.format(s_name, GO_term))
            query_values.append((s_name, GO_term))
    run(query_values)

if __name__ == '__main__':
    # retrieve_ensembl_ids('Data_retriever/downloaded_data/9606.protein.actions.v11.0.txt', 'Data_retriever/downloaded_data/ensembl_ids.txt')
    # write_human_gene_queries('Data_retriever/downloaded_data/ensembl_ids.txt', 'Data_retriever/queries_folder/human_gene_queries.txt')
    write_human_go_queries('Data_retriever/downloaded_data/uniprot_ensemble_GO_term.csv')