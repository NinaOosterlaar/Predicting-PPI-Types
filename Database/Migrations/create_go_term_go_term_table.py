create_commands = (
    """
        CREATE TABLE IF NOT EXISTS go_term_go_term (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            go_id_parent VARCHAR(255) NOT NULL REFERENCES go_term(id) ON DELETE CASCADE ON UPDATE CASCADE,
            go_id_child VARCHAR(255) NOT NULL REFERENCES go_term(id) ON DELETE CASCADE ON UPDATE CASCADE,
            go_term_relation_id INTEGER NOT NULL REFERENCES go_term_relation(id) ON DELETE CASCADE ON UPDATE CASCADE
        )
    """,
)

destroy_commands = (
    """DROP TABLE IF EXISTS go_term_go_term CASCADE""",
)

def get_create_commands():
    return create_commands

def get_destroy_commands():
    return destroy_commands