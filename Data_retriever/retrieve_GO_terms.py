import os

def retrieve_GO_terms():
    """Retrieves the GO terms from the obo file
        and returns a dictionary with the GO id as the key and a dictionary
        with the term, category, and relationships
        return value: a dictionary with the GO id as the key and a dictionary"""
    GO_terms  = {}
    
    with open("Data_retriever\go-basic.obo", "r") as f:
        data = f.read().split("[Term]\n")[1:]
        most_data = data[:-1]
        
        for term in most_data:
            GO_terms.update(parse_GO_term(term))
        
        last_term = data[-1].split("[Typedef]\n")[0]
        GO_terms.update(parse_GO_term(last_term))
    
    return GO_terms


def parse_GO_term(GO_term: str):
    """Parses a GO term from the obo file and returns a dictionary with the 
        GO id as the key and a dictionary with the term, category, and relationships
        GO_term: a string containing the GO term from the obo file
        return value: a dictionary with the GO id as the key and a dictionary with the term, category, and relationships"""
    terms = {}
    GO_term = GO_term.split("\n")
    
    id = GO_term[0][4:]
    term = GO_term[1][6:]
    category = GO_term[2][11:]
    
    terms[id] = {"term": term, "category": category, "is_a": [], "part_of": [],
                 "regulates": [], "positively_regulates": [], "negatively_regulates": []}
    
    for line in GO_term[3:]:
        if line.startswith("is_a:"):
            terms[id]["is_a"].append(line[6:16])
        elif line.startswith("relationship: part_of"):
            terms[id]["part_of"].append(line[22:32])
        elif line.startswith("relationship: regulates"):
            terms[id]["regulates"].append(line[24:34])
        elif line.startswith("relationship: positively_regulates"):
            terms[id]["positively_regulates"].append(line[35:45])
        elif line.startswith("relationship: negatively_regulates"):
            terms[id]["negatively_regulates"].append(line[35:45])
    
    return terms


def create_queries_go_term():
    """Creates the queries to insert the GO terms into the database"""
    terms = retrieve_GO_terms()
    filename = "Data_retriever\GO_term_queries.txt"
    if os.path.exists(filename):
        os.remove(filename)
    for key, value in terms.items():
        category_id = None
        if(value["category"] == "biological_process"):
            category_id = 1
        elif(value["category"] == "molecular_function"):
            category_id = 2
        elif(value["category"] == "cellular_component"):
            category_id = 3
            
        term = value["term"].replace("'", "''")

        with open(filename, "a") as f:
            f.write('"INSERT INTO go_term values (\'{}\', \'{}\', {});",\n'.format(key, term, category_id))


def create_quries_go_term_go_term():
    """
    This function will retrieve all the parent/child relationships from the GO terms
    """
    terms = retrieve_GO_terms()
    filename = "Data_retriever\GO_term_GO_term_queries.txt"
    if os.path.exists(filename):
        os.remove(filename)
    for key, value in terms.items():
        for relation in value:
            relation_number = None
            if(relation == "is_a"):
                relation_number = 1
                write_go_term_go_term_query(relation_number, value[relation], key, filename)
            elif(relation == "part_of"):
                relation_number = 2
                write_go_term_go_term_query(relation_number, value[relation], key, filename)
            elif(relation == "regulates"):
                relation_number = 3
                write_go_term_go_term_query(relation_number, value[relation], key, filename)
            elif(relation == "positively_regulates"):
                relation_number = 4
                write_go_term_go_term_query(relation_number, value[relation], key, filename)
            elif(relation == "negatively_regulates"):
                relation_number = 5
                write_go_term_go_term_query(relation_number, value[relation], key, filename)
                

def write_go_term_go_term_query(number: int, go_ids: list, go_id: str, filename: str):
    for id in go_ids:
        with open(filename, "a") as f:
            f.write('"INSERT INTO go_term_go_term (go_id_parent, go_id_child, go_term_relation_id) values (\'{}\', \'{}\', {});",\n'.format(id, go_id, number))
            
    
if __name__ == "__main__":
    create_quries_go_term_go_term()