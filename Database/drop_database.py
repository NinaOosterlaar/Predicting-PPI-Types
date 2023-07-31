from Migrations import destroy_migrations
from run_command import run
            
if __name__ == '__main__':
    for migration in destroy_migrations:
            run(migration, batch=True)