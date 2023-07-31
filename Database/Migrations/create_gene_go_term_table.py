create_commands = (
    """
        CREATE TABLE IF NOT EXISTS gene_go_term (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            s_name VARCHAR(255) NOT NULL REFERENCES gene(s_name) ON DELETE CASCADE ON UPDATE CASCADE,
            go_id VARCHAR(255) NOT NULL REFERENCES go_term(id) ON DELETE CASCADE ON UPDATE CASCADE,
            qualifier_id INTEGER REFERENCES qualifier(id) ON DELETE CASCADE ON UPDATE CASCADE
        )
    """,
)

destroy_commands = (
    """DROP TABLE IF EXISTS gene_go_term CASCADE""",
)

def get_create_commands():
    return create_commands

def get_destroy_commands():
    return destroy_commands