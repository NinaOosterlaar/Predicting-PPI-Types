# This is an automatically generated script to run your query
# to use it you will require the intermine python client.
# To install the client, run the following command from a terminal:
#
#     sudo easy_install intermine
#
# For further documentation you can visit:
#     http://intermine.readthedocs.org/en/latest/web-services/

# The line below will be needed if you are running this script with python 2.
# Python 3 will ignore it.
from __future__ import print_function
from urllib.parse import urlparse
import os

# The following two lines will be needed in every python script:
from intermine.webservice import Service

def run_yeastmine_genes():
    """
    This function will run a query to retrieve all the genes from yeastmine
    """
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    filename = "Data_retriever/yeast_genes_queries.txt"
    if(os.path.exists(filename)):
        os.remove(filename)

    # Get a new query on the class (table) you will be querying:
    query = service.new_query("Gene")

    # The view specifies the output columns
    query.add_view("secondaryIdentifier", "symbol")

    # You can edit the constraint values below
    query.add_constraint("organism.shortName", "=", "S. cerevisiae", code="A")

    # Uncomment and edit the code below to specify your own custom logic:
    # query.set_logic("A")

    for row in query.rows():
        with open(filename, "a") as myfile:
            myfile.write('"INSERT INTO gene values (\'{}\', {}, {});",\n'.format(row["secondaryIdentifier"], 1, '\'' + row["symbol"] + '\'' if row["symbol"] else "NULL"))


def run_yeastmine_gene_go():
    """
    This function will run a query to retrieve all the GO terms from all the genes from yeastmine
    """
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    filename = "Data_retriever/yeast_genes_GO_queries.txt"
    if(os.path.exists(filename)):
        os.remove(filename)
        
    map_qualfier = {'involved in': 1, 'enables': 2, 'part of': 3, 'is active in': 4, 'contributes to': 5, 'acts upstream of or within': 6, 'located in': 7, 'colocalizes with': 8, 'NOT': 9, 'acts upstream of or within positive effect': 10, 'acts upstream of positive effect': 11, 'acts upstream of': 12, 'acts upstream of negative effect': 13}

    # Get a new query on the class (table) you will be querying:
    query = service.new_query("Gene")

    # The view specifies the output columns
    query.add_view(
        "secondaryIdentifier", "goAnnotation.ontologyTerm.identifier",
        "goAnnotation.qualifier"
    )

    # You can edit the constraint values below
    query.add_constraint("organism.shortName", "=", "S. cerevisiae", code="F")
    query.add_constraint("status", "IS NULL", code="C")
    query.add_constraint("status", "=", "Active", code="B")
    query.add_constraint("Gene", "IN", "Verified_ORFs", code="A")

    # Your custom constraint logic is specified with the code below:
    query.set_logic("(B or C) and F and A")

    for row in query.rows():
        with open(filename, "a") as myfile:
            myfile.write('"INSERT INTO gene_go_term (s_name, qualifier_id, go_id) values (\'{}\', {}, {});",\n'.format(row["secondaryIdentifier"], map_qualfier[row["goAnnotation.qualifier"]], '\'' + row["goAnnotation.ontologyTerm.identifier"] + '\''))
            


if __name__ == '__main__':
    run_yeastmine_gene_go()
 