create_commands = (
    """
        CREATE TABLE IF NOT EXISTS gene (
            s_name VARCHAR(255) PRIMARY KEY,
            organism_id INTEGER NOT NULL REFERENCES organism(id) ON DELETE CASCADE ON UPDATE CASCADE,
            r_name VARCHAR(255)
        )
    """,
)

destroy_commands = (
    """DROP TABLE IF EXISTS gene CASCADE""",
)

def get_create_commands():
    return create_commands

def get_destroy_commands():
    return destroy_commands