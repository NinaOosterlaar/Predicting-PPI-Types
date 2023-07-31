from .go_term_relation_seeder import get_seeds as go_term_relation_seeds
from .go_term_relation_seeder import destroy_seeds as destroy_go_term_relation_seeds
from .go_term_category_seeder import get_seeds as go_term_category_seeds
from .go_term_category_seeder import destroy_seeds as destroy_go_term_category_seeds
from .go_term_seeder import get_seeds as go_term_seeds
from .go_term_seeder import destroy_seeds as destroy_go_term_seeds
from .go_term_go_term_seeder import get_seeds as go_term_go_term_seeds
from .go_term_go_term_seeder import destroy_seeds as destroy_go_term_go_term_seeds
from .organism_seeder import get_seeds as organism_seeds
from .organism_seeder import destroy_seeds as destroy_organism_seeds
from .gene_seeder import get_seeds as gene_seeds
from .gene_seeder import destroy_seeds as destroy_gene_seeds
from .go_qualifier_seeder import get_seeds as go_qualifier_seeds
from .go_qualifier_seeder import destroy_seeds as destroy_go_qualifier_seeds
from .gene_go_seeder import get_seeds as gene_go_seeds
from .gene_go_seeder import destroy_seeds as destroy_gene_go_seeds
from .ppi_type_seeder import get_seeds as ppi_type_seeds
from .ppi_type_seeder import destroy_seeds as destroy_ppi_type_seeds
from .ppi_seeder import get_seeds as ppi_seeds
from .ppi_seeder import destroy_seeds as destroy_ppi_seeds
from .duplicate_removed import get_seeds as duplicate_removed_seeds

seeds = [
    # go_term_relation_seeds,
    # go_term_category_seeds,
    # go_term_seeds,
    # go_term_go_term_seeds,
    # organism_seeds,
    # gene_seeds,
    # go_qualifier_seeds,
    # gene_go_seeds,
    # ppi_type_seeds,
    ppi_seeds,
    # duplicate_removed_seeds,
]

destroy_seeds = [
    destroy_go_term_relation_seeds,
    destroy_go_term_category_seeds,
    destroy_go_term_seeds,
    destroy_go_term_go_term_seeds,
    destroy_organism_seeds,
    destroy_gene_seeds,
    destroy_go_qualifier_seeds,
    destroy_gene_go_seeds,
    destroy_ppi_type_seeds,
    destroy_ppi_seeds,
]