# contexual_purs
A implementation of PURS with contextual data.

Dependency:
if cpu only:
pip install tensorflow==1.14

if use gpu:
pip install tensorflow-gpu==1.14
conda install cudatoolkit

example command line:

cd /path/to/project/code

python train_purs.py --device /gpu:4 --batch-size 1000 --dataset jester

device = {/cpu:0, /gpu:x}

dataset = {beer, beer_small, beer_ultrasmall, jester, jester_small}
