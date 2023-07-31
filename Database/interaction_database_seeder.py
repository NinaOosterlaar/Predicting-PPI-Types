import csv
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
        args_str = ','.join(["('{}','{}',{},{},'{}','{}')".format(*x) for x in parameter])
        cur.execute("INSERT OR IGNORE INTO ppi (s_name_1, s_name_2, ppi_type_id, confidence_score, is_directional, one_is_acting) VALUES " + args_str)
        conn.commit()
        cur.close()
    except Error as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def run2(parameter, select = False, batch = False):
    conn = create_connection(database_path)
    result = None
    try:
        cur = conn.cursor()
        if batch:
            for command in parameter():
                # print("Executing: " + command + "\n")
                cur.execute(command)   
        else:    
            cur.execute(parameter)
        
        if select:
            result = cur.fetchall()  # Retrieve the result
        conn.commit()
        cur.close()
    except Error as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return result

def create_queries_yeast_interactinos(filename, human):                 
    query_values = []
    with open(filename, 'r') as f:
        interactions = list(csv.reader(f, delimiter='\t'))[1:]
    
    if human:
        with open("Data_retriever\downloaded_data\ensembl_ids.txt") as f:
            ensembl_ids = frozenset(f.read().splitlines())
    
    for interaction in interactions:
        if human:
            if interaction[0][5:] not in ensembl_ids or interaction[1][5:] not in ensembl_ids:
                # print(interaction[0][5:], interaction[1][5:])
                continue
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
        query_values.append((interaction[0][7:], interaction[1][7:], ppi_type, interaction[-1], interaction[-3], interaction[-2]))
        # f.write('"INSERT INTO ppi (s_name_1, s_name_2, ppi_type_id, confidence_score, is_directional, one_is_acting) values (\'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\');",\n'.format(interaction[0][5:], interaction[1][5:], ppi_type, interaction[-1], interaction[-3], interaction[-2]))
    # print(len(query_values))
    run(query_values)
    
    
def retrieve_interactions(ppi_type_id, confidence_score): 
    return run2(f"""
        SELECT s_name_1, s_name_2 
        FROM ppi
        WHERE ppi_type_id = {ppi_type_id}
        AND confidence_score > {confidence_score};
        """, True)
    

def search_interaction_actionless():
    pos_interaction = retrieve_interactions(1, 0)
    neg_interaction = retrieve_interactions(2, 0)
    pos_and_neg = frozenset(pos_interaction + neg_interaction)
    not_action = set()
    for type in range(3, 8):
        print(type) 
        interactions = retrieve_interactions(type, 0)
        for interaction in interactions:
            if interaction not in pos_and_neg:
                not_action.add(interaction)
    not_action = set(not_action)
    query_values = []   
    for interaction in not_action:
        query_values.append((interaction[0], interaction[1], 9, 0, False, False))
    print(len(not_action))
    print(len(pos_and_neg))
    print(len(not_action.intersection(pos_and_neg)))
    run(query_values)
    

def retrieve_organism_interactions(ppi_type_id, confidence_score, organism_id):
    return run2(f"""
        SELECT s_name_1, s_name_2 
        FROM ppi, gene
        WHERE ppi.ppi_type_id = {ppi_type_id}
        AND (ppi.s_name_1 = gene.s_name OR ppi.s_name_2 = gene.s_name)
        AND ppi.confidence_score >= {confidence_score}
        AND gene.organism_id = {organism_id};
        """, True)
    
                
if __name__ == '__main__':
    create_queries_yeast_interactinos(r"Data_retriever\downloaded_data\511145.protein.actions.v11.0.txt", False)
    # search_interaction_actionless()
    # conn = create_connection(database_path)
    # try:
    #     cur = conn.cursor()
    #     # args_str = ','.join(["('{}','{}',{},{},'{}','{}')".format(*x) for x in query_values])
    #     cur.execute("DELETE FROM ppi, gene WHERE (ppi.s_name_1 = gene.s_name OR ppi.s_name_2 = gene.s_name) AND gene.organism_id = 3")
    #     conn.commit()
    #     cur.close()
    # except Error as error:
    #     print(error)
    # finally:
    #     if conn is not None:
    #         conn.close() 