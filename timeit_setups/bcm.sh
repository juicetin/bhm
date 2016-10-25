SETUP="
from ML.gp.bcm import BCM

bcm = BCM()
"

python -m timeit -s "$SETUP"

