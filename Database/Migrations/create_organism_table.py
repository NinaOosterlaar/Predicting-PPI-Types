create_commands = (
    """
        CREATE TABLE IF NOT EXISTS organism (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(255) NOT NULL
        )
    """,
)

destroy_commands = (
    """DROP TABLE IF EXISTS organism CASCADE""",
)

def get_create_commands():
    return create_commands

def get_destroy_commands():
    return destroy_commands