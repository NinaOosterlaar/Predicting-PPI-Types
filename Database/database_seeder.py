from Seeders import seeds
from run_command import run
            
if __name__ == '__main__':
    for seed in seeds:
        run(seed, batch=True)
        print('Migration seeded.')