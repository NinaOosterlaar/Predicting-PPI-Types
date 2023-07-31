from Migrations import create_migrations
from run_command import run
            
if __name__ == '__main__':
    for migration in create_migrations:
        run(migration, batch=True)