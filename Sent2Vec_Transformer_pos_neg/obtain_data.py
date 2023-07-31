import sqlite3 as sq
from sqlite3 import Error

"""Change database path to the path of the database you want to use"""
database_path = r"C:\Users\\ninao\OneDrive - Delft University of Technology\Nanobiology\Bachelor End Project\Bep\Database\database.sqlite"

def create_connection(db_file):
    try:
        return sq.connect(db_file)
    except Error as e:
        print(e)
    
    return None

def run(parameter, select = True, batch = False):
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
            result = cur.fetchall()
        conn.commit()
        cur.close()
    except Error as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return result


def retrieve_interactions(ppi_type_id, confidence_score): 
    return run(f"""
        SELECT s_name_1, s_name_2 
        FROM ppi
        WHERE ppi_type_id = {ppi_type_id}
        AND confidence_score > {confidence_score};
        """)
        

def retrieve_gene_GO_term(organism_id = 0):
    if organism_id == 0:
        return run("""
                SELECT s_name, go_id
                FROM gene_go_term;
                """)
    else:
        return run(f"""
                SELECT s_name, go_id
                FROM gene_go_term
                WHERE organism_id = {organism_id};
                """)


def create_gene_GO_dict(organism_id = 0):
    result = retrieve_gene_GO_term(organism_id)
    gene_GO_dict = {}
    for gene, GO_term in result:
        if gene in gene_GO_dict and GO_term not in gene_GO_dict[gene]:
            gene_GO_dict[gene].append(GO_term)
        else:
            gene_GO_dict[gene] = [GO_term]
    return gene_GO_dict


def delete_duplicates(interactions_pos, interactions_neg):
    pos_set = frozenset(retrieve_interactions(1, 0))
    neg_set = frozenset(retrieve_interactions(2, 0))
    
    new_pos_interactions = []
    for interaction in interactions_pos:
        if interaction not in neg_set:
            new_pos_interactions.append(interaction)
            
    new_neg_interactions = []
    for interaction in interactions_neg:
        if interaction not in pos_set:
            new_neg_interactions.append(interaction)

    return new_pos_interactions, new_neg_interactions



def count_duplicates(pos, neg):
    print('total positive: ',len(pos))
    print('total negative: ',len(neg))
    pos_set = set(pos)
    print('unique positive: ',len(pos_set))
    neg_set = set(neg)
    print('unique negative: ',len(neg_set))
    both = pos_set.intersection(neg_set)
    print('both pos and neg: ',len(both))
    
    
def retrieve_confidence_score(ppi_type_id, s_name_1, s_name_2): 
    return run(f"""
        SELECT confidence_score
        FROM ppi
        WHERE ppi_type_id = {ppi_type_id}
        AND s_name_1 = '{s_name_1}'
        AND s_name_2 = '{s_name_2}';
        """)
    
    
def retrieve_organism_interactions(ppi_type_id, confidence_score, organism_id):
    return run(f"""
        SELECT s_name_1, s_name_2 
        FROM ppi, gene
        WHERE ppi.ppi_type_id = {ppi_type_id}
        AND (ppi.s_name_1 = gene.s_name OR ppi.s_name_2 = gene.s_name)
        AND ppi.confidence_score >= {confidence_score}
        AND gene.organism_id = {organism_id};
        """)

            
if __name__ == '__main__':
    # # pass
    # pos_interactions = retrieve_interactions(1, 900)
    # neg_interactions = retrieve_interactions(2, 700)
    # # # print(len(set(pos_interactions).intersection(set(neg_interactions))))
    # print(len(pos_interactions), len(neg_interactions))
    # # pos_interactions, neg_interactions = delete_duplicates(pos_interactions, neg_interactions)
    # # # print(len(pos_interactions), len(neg_interactions)) 
    # # # print(len(set(pos_interactions).intersection(set(neg_interactions))))
    # # print(len(pos_interactions))
    # # print(len(neg_interactions))
    # # print(len(set(pos_interactions)))
    # # print(len(set(neg_interactions)))
    # # print(len(pos_interactions) - len(set(pos_interactions)))
    # # print(len(neg_interactions) - len(set(neg_interactions)))
    print(len(retrieve_organism_interactions(1, 700, 3)))
    print(len(retrieve_organism_interactions(2, 700, 3)))
    