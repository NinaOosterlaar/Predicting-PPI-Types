import os
import csv


def get_genes(filename, output_file):                
    with open(filename, 'r') as f:
        interactions = list(csv.reader(f, delimiter='\t'))[1:]
        
    genes = []
    for interaction in interactions:
        gene1 = interaction[0].split('.')[1]
        gene2 = interaction[1].split('.')[1]
        if(gene1 not in genes):
            genes.append(gene1)
        if(gene2 not in genes):
            genes.append(gene2)
    create_query_file(output_file, genes)   
    


def create_query_file(output_file, genes):
    if(os.path.exists(output_file)):
        os.remove(output_file)
    
    for gene in genes:
        with open(output_file, 'a') as f:
            f.write('"INSERT OR IGNORE INTO gene (s_name, organism_id) values (\'{}\', \'{}\');",\n'.format(gene, 3))


def create_gene_go_terms_queries(input_file, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)
    
    with open(input_file, 'r') as f:
        lines = f.readlines()[32:]
    
    for line in lines:
        line = line.split('\t')
        gene = line[10].split('|')[-1]
        go_term = line[4]
        with open(output_file, 'a') as f:
            f.write('"INSERT OR IGNORE INTO gene_go_term (s_name, go_id) values (\'{}\', \'{}\');",\n'.format(gene, go_term))


if __name__ == '__main__':
    # get_genes("Data_retriever\downloaded_data\\511145.protein.actions.v11.0.txt", "Data_retriever\queries_folder\e_coli_genes_queries.txt")
    create_gene_go_terms_queries("Data_retriever\downloaded_data\ecocyc.gaf", "Data_retriever\queries_folder\e_coli_gene_go_terms_queries.txt")