import sqlite3 as sq
from sqlite3 import Error
import pickle as pkl

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


def retrieve_gene_GO_term(organism_id = 0):
    if organism_id == 0:
        return run("""
                SELECT s_name, go_id
                FROM gene_go_term;
                """)
    else:
        return run(f"""
                SELECT gene_go_term.s_name, go_id
                FROM gene_go_term, gene
                WHERE gene.s_name = gene_go_term.s_name
                AND gene.organism_id = {organism_id};
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


def count_go_terms_type(filename,filename2):
    with open(filename, 'rb') as f:
        proteins = pkl.load(f)
        
    with open(filename2, 'rb') as f:
        proteins2 = pkl.load(f)
        
        
    gene_GO_dict = create_gene_GO_dict()
    
    help_cry = proteins2['correct_1']
    
        
    
    correct_0 = proteins['correct_0']
    correct_1 = proteins['correct_1']
    correct_2 = proteins['correct_2']
    false_0_is_1 = proteins['false_0_is_1']
    false_1_is_0 = proteins['false_1_is_0']
    false_0_is_2 = proteins['false_0_is_2']
    false_1_is_2 = proteins['false_1_is_2']
    false_2_is_0 = proteins['false_2_is_0']
    false_2_is_1 = proteins['false_2_is_1']
    total = correct_0 + correct_1 + correct_2 + false_0_is_1 + false_1_is_0 + false_0_is_2 + false_1_is_2 + false_2_is_0 + false_2_is_1
    
    correct_0_go = {}
    correct_1_go = {}
    correct_2_go = {}
    false_0_is_1_go = {}
    false_0_is_2_go = {}
    false_1_is_0_go = {}
    false_1_is_2_go = {}
    false_2_is_0_go = {}
    false_2_is_1_go = {}
    
    # print(len(total))
    
    # count = 0
    # for protein in help_cry:
    #     print(protein)
    #     if protein in correct_1:
    #         print(protein)
    #         count += 1
    # print(count)
    # print(count/len(help_cry)) 
    # print(len(help_cry)) 
    # print(len(correct_1))
    
    # print(len(gene_GO_dict.keys()))
    # proteins_retrieve_go(false_2_is_1, {}, gene_GO_dict)
    # proteins_retrieve_go(correct_1, correct_1_go, gene_GO_dict)
    
    # proteins_retrieve_go(total, {}, gene_GO_dict)
    proteins_retrieve_go(correct_0, correct_0_go, gene_GO_dict)
    proteins_retrieve_go(correct_1, correct_1_go, gene_GO_dict)
    proteins_retrieve_go(correct_2, correct_2_go, gene_GO_dict)
    # proteins_retrieve_go(false_0_is_1, false_0_is_1_go, gene_GO_dict)
    # proteins_retrieve_go(false_0_is_2, false_0_is_2_go, gene_GO_dict)
    # proteins_retrieve_go(false_1_is_0, false_1_is_0_go, gene_GO_dict)
    proteins_retrieve_go(false_1_is_2, false_1_is_2_go, gene_GO_dict)
    # proteins_retrieve_go(false_2_is_0, false_2_is_0_go, gene_GO_dict)
    # proteins_retrieve_go(false_2_is_1, false_2_is_1_go, gene_GO_dict)

    
    count = 0
    for GO in false_1_is_2_go.keys():
        if GO in correct_0_go.keys():
            count += 1
    print(count)
    print(count/len(false_1_is_2_go.keys()))
    
    # count = 0
    # for protein_inter in false_1:
    #     if protein_inter in false_1_sent:
    #         count += 1
    # print(count)
    # print(len(false_1))
    # print(count/len(false_1))
    
    
    
    
def proteins_retrieve_go(proteins_list, go_dic, gene_GO_dict):
    count = 0
    GO_term_set = set()
    # print(len(proteins_list))
    for proteins in proteins_list:
        for protein in proteins:
            # print(protein)
            if protein in gene_GO_dict:
                # print(len(gene_GO_dict[protein]))
                for GO_term in gene_GO_dict[protein]:
                    GO_term_set.add(GO_term)
                    if GO_term not in go_dic:
                        count += 1
                        go_dic[GO_term] = 1
                    else:
                        go_dic[GO_term] += 1
    max_val = 0
    total_frequencies = 0
    for GO_term in go_dic.keys():
        total_frequencies += go_dic[GO_term]
    # print(total_frequencies)
    # print(total_frequencies/len(proteins_list))
    # print(len(GO_term_set))

    
        

def overlap_organism_GO():
    yeast =create_gene_GO_dict(organism_id=1)
    human = create_gene_GO_dict(organism_id=2)
    ecoli = create_gene_GO_dict(organism_id=3)
    pombe = create_gene_GO_dict(organism_id=4)
    
    training_GO = set()
    ecoli_GO = set()
    pombe_GO = set()
    
    for protein in yeast.keys():
        for GO_term in yeast[protein]:
            training_GO.add(GO_term)
    for protein in human.keys():
        for GO_term in human[protein]:
            training_GO.add(GO_term)
    
    for protein in ecoli.keys():
        for GO_term in ecoli[protein]:
            ecoli_GO.add(GO_term)
    
    for protein in pombe.keys():
        for GO_term in pombe[protein]:
            pombe_GO.add(GO_term)
    
    print(len(ecoli_GO))
    print(len(pombe_GO))
    count_ecoli = 0
    count_pombe = 0
    
    for GO_term in ecoli_GO:
        if GO_term in training_GO:
            count_ecoli += 1
    for GO_term in pombe_GO:
        if GO_term in training_GO:
            count_pombe += 1
    
    print(count_ecoli)
    print(count_pombe)
    
    print(count_ecoli/len(ecoli_GO))
    print(count_pombe/len(pombe_GO))
    

    
    
if __name__ == '__main__':
    # count_go_terms_type(r'results\proteins_yeah_node2vec.pkl', r'results\proteins_yeah_sent2vec.pkl')
    # overlap_organism_GO()
    # with open(r'results\Node2Vec\multiclass\proteins_yeah.pkl', 'rb') as f:
    #     proteins = pkl.load(f)
    # print(proteins.keys())
    count_go_terms_type(r'results\proteins_yeah_mul_sent2ve.pkl', r'results\proteins_yeah_node2vec.pkl')