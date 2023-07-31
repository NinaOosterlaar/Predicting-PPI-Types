seeds = (
    "INSERT OR IGNORE INTO qualifier VALUES (1, 'involved in');",
    "INSERT OR IGNORE INTO qualifier VALUES (2, 'enables');",
    "INSERT OR IGNORE INTO qualifier VALUES (3, 'part of');",
    "INSERT OR IGNORE INTO qualifier VALUES (4, 'is active in');",
    "INSERT OR IGNORE INTO qualifier VALUES (5, 'contributes to');",
    "INSERT OR IGNORE INTO qualifier VALUES (6, 'acts upstream of or within');",
    "INSERT OR IGNORE INTO qualifier VALUES (7, 'located in');",
    "INSERT OR IGNORE INTO qualifier VALUES (8, 'colocalizes with');",
    "INSERT OR IGNORE INTO qualifier VALUES (9, 'NOT');",
    "INSERT OR IGNORE INTO qualifier VALUES (10, 'acts upstream of or within positive effect');",
    "INSERT OR IGNORE INTO qualifier VALUES (11, 'acts upstream of positive effect');",
    "INSERT OR IGNORE INTO qualifier VALUES (12, 'acts upstream of');",
    "INSERT OR IGNORE INTO qualifier VALUES (13, 'acts upstream of negative effect');",
)

def get_seeds():  
    return seeds

def destroy_seeds():
    return ("DELETE FROM qualifier;",)