seeds = (
    "INSERT OR IGNORE INTO organism VALUES (1, 'saccharomyces_cerevisiae');",
    "INSERT OR IGNORE INTO organism VALUES (2, 'homo_sapiens');",
    "INSERT OR IGNORE INTO organism VALUES (3, 'Escherichia coli')",
    "INSERT OR IGNORE INTO organism VALUES (4, 'Schizosaccharomyces pombe')"
)

def get_seeds():  
    return seeds

def destroy_seeds():
    return ("DELETE FROM organism;",)
