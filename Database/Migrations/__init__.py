from .create_go_term_category_table import get_create_commands as create_go_term_category_table
from .create_go_term_category_table import get_destroy_commands as destroy_go_term_category_table
from .create_go_term_table import get_create_commands as create_go_term_table
from .create_go_term_table import get_destroy_commands as destory_go_term_table
from .create_go_term_relation_table import get_create_commands as create_go_term_relation_table
from .create_go_term_relation_table import get_destroy_commands as destroy_go_term_relation_table
from .create_go_term_go_term_table import get_create_commands as create_go_term_go_term_table
from .create_go_term_go_term_table import get_destroy_commands as destroy_go_term_go_term_table
from .create_organism_table import get_create_commands as create_organism_table
from .create_organism_table import get_destroy_commands as destroy_organism_table
from .create_gene_table import get_create_commands as create_gene_table
from .create_gene_table import get_destroy_commands as destroy_gene_table
from .create_ppi_type_table import get_create_commands as create_ppi_type_table
from .create_ppi_type_table import get_destroy_commands as destroy_ppi_type_table
from .create_ppi_table import get_create_commands as create_ppi_table
from .create_ppi_table import get_destroy_commands as destroy_ppi_table
from .create_gene_go_term_table import get_create_commands as create_gene_go_term_table
from .create_gene_go_term_table import get_destroy_commands as destroy_gene_go_term_table
from .create_qualifier_table import get_create_commands as create_qualifier_table
from .create_qualifier_table import get_destroy_commands as destroy_qualifier_table

create_migrations = [
    create_go_term_category_table,
    create_qualifier_table,
    create_go_term_table,
    create_go_term_relation_table,
    create_go_term_go_term_table,
    create_organism_table,
    create_gene_table,
    create_ppi_type_table,
    create_ppi_table,
    create_gene_go_term_table,
]

destroy_migrations = [
    destroy_go_term_category_table,
    destroy_qualifier_table,
    destory_go_term_table,
    destroy_go_term_relation_table,
    destroy_go_term_go_term_table,
    destroy_organism_table,
    destroy_gene_table,
    destroy_ppi_type_table,
    destroy_ppi_table,
    destroy_gene_go_term_table,
]