from os.path import dirname, basename, join
import numpy as np
EXPERIMENT_DIR = dirname(__file__)

def main():
    lrs = 10**np.random.uniform(-1, 1, size = 10)

    for i, lr in enumerate(lrs):      
        save_path = f'/scratch/nw1045/contexual_purs/jester_baseline/{i}_lr_{lr}'
        with open(join("config.sh"), "r") as f:
            TEMPLATE = f.read()
            exp_config_path = join(EXPERIMENT_DIR, f'jester_baseline_{i}.sh')

            with open(exp_config_path, "w") as c:

                tt = TEMPLATE.replace("$lr$", str(lr))
                tt = tt.replace("$savepath$", save_path)
                c.write(tt)

if __name__ == '__main__':
    
    main()