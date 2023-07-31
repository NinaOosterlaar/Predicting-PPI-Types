from Seeders import destroy_seeds
from run_command import run
            
if __name__ == '__main__':
    for seed in destroy_seeds:
            run(seed, batch=True)