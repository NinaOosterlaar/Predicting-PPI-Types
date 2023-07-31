create_commands = (
    """
        CREATE TABLE IF NOT EXISTS ppi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            s_name_1 VARCHAR(255) NOT NULL REFERENCES gene(s_name) ON DELETE CASCADE ON UPDATE CASCADE,
            s_name_2 VARCHAR(255) NOT NULL REFERENCES gene(s_name) ON DELETE CASCADE ON UPDATE CASCADE,
            ppi_type_id INTEGER NOT NULL REFERENCES ppi_type(id) ON DELETE CASCADE ON UPDATE CASCADE,
            confidence_score FLOAT NOT NULL,
            is_directional BOOLEAN NOT NULL,
            one_is_acting BOOLEAN NOT NULL
        )
    """,
)

destroy_commands = (
    """DROP TABLE IF EXISTS ppi CASCADE""",
)

def get_create_commands():
    return create_commands

def get_destroy_commands():
    return destroy_commands