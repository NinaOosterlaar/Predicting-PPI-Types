seeds = (
    "INSERT OR IGNORE INTO go_term_category values (1, 'biological_process');",
    "INSERT OR IGNORE INTO go_term_category values (2, 'molecular_function');",
    "INSERT OR IGNORE INTO go_term_category values (3, 'cellular_component');"
)

def get_seeds():  
    return seeds

def destroy_seeds():
    return ("DELETE FROM go_term_category;",)