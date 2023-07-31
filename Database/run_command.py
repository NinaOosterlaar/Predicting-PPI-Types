import sqlite3 as sq
from sqlite3 import Error
# from Seeders.organism_seeder import get_seeds as organism_seeds

retrieve_go_terms = """
                        SELECT s_name, term, c.name 
                        FROM gene_go_term as g
                        INNER JOIN go_term AS t ON t.id = g.go_id
                        INNER JOIN go_term_category AS c ON c.id = t.go_term_category_id
                        """

database_path = r"C:\Users\\ninao\OneDrive - Delft University of Technology\Nanobiology\Bachelor End Project\Bep\Database\database.sqlite"

def create_connection(db_file):
    try:
        return sq.connect(db_file)
    except Error as e:
        print(e)
    
    return None

def run(parameter, select = False, batch = False):
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
        conn.commit()
        cur.close()
    except Error as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return result


def create_dictioanry_go_terms(gene_go_terms: list):
    result = {}
    for gene_go_term in gene_go_terms:
        if(gene_go_term[0] not in result):
            result[gene_go_term[0]] = {'biological_process': [], 'cellular_component': [], 'molecular_function': []}
        if(gene_go_term[1] != gene_go_term[2]):
            result[gene_go_term[0]][gene_go_term[2]].append(gene_go_term[1])
    return result


def average_go_terms(gene_go_dict: dict):
    count = 0
    total = 0
    for gene in gene_go_dict:
        for category in gene_go_dict[gene]:
            total += len(gene_go_dict[gene][category])
            count += 1
    return total/count


def retrieve_gene_GO_term():
    gene_go_term = run("""
        SELECT s_name, go_id
        FROM gene_go_term 
        """, True)
    return gene_go_term


def make_file(filename):
    with open(filename, 'w') as f:
        for gene_go_term in retrieve_gene_GO_term():
            f.write(f'{gene_go_term[0]} {gene_go_term[1]}\n')
            
            
if __name__ == '__main__':
    # temp  = run(retrieve_go_terms, True)
    # result = create_dictioanry_go_terms(temp)
    # print(average_go_terms(result))
    make_file('gene_go_term.txt')
    