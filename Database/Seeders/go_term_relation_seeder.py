seeds = (
    "INSERT OR IGNORE INTO go_term_relation values (1, 'is_a');",
    "INSERT OR IGNORE INTO go_term_relation values (2, 'part_of');",
    "INSERT OR IGNORE INTO go_term_relation values (3, 'regulates');",
    "INSERT OR IGNORE INTO go_term_relation values (4, 'positively_regulates');",
    "INSERT OR IGNORE INTO go_term_relation values (5, 'negatively_regulates');"
)

def get_seeds():  
    return seeds

def destroy_seeds():
    return ("DELETE FROM go_term_relation;",)