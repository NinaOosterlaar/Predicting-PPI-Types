seeds = (
    "INSERT OR IGNORE INTO ppi_type VALUES (1, 'activation');",
    "INSERT OR IGNORE INTO ppi_type VALUES (2, 'inhibition');",
    "INSERT OR IGNORE INTO ppi_type VALUES (3, 'binding');",
    "INSERT OR IGNORE INTO ppi_type VALUES (4, 'catalysis');",
    "INSERT OR IGNORE INTO ppi_type VALUES (5, 'ptmod');",
    "INSERT OR IGNORE INTO ppi_type VALUES (6, 'reaction');",
    "INSERT OR IGNORE INTO ppi_type VALUES (7, 'expression');",
    "INSERT OR IGNORE INTO ppi_type VALUES (8, 'else');",
    "INSERT OR IGNORE INTO ppi_type VALUES (9, 'no action');"
)

def get_seeds():  
    return seeds

def destroy_seeds():
    return ("DELETE FROM ppi_type;",)