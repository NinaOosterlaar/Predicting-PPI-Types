create_commands = (
    """
        CREATE TABLE IF NOT EXISTS go_term (
            id VARCHAR(255) PRIMARY KEY,
            term VARCHAR(255) NOT NULL,
            go_term_category_id INTEGER NOT NULL REFERENCES go_term_category(id) ON DELETE CASCADE ON UPDATE CASCADE
        )
    """,
)

destroy_commands = (
    """DROP TABLE IF EXISTS go_term CASCADE""",
)

def get_create_commands():
    return create_commands

def get_destroy_commands():
    return destroy_commands