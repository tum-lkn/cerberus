# Flow Simulator
Python-based flow level simulator for Cerberus: The Power of Choices in Datacenter TopologyDesign by Griner et al. 

Build the docker image
```bash
docker-compose -f docker/docker-compose.ym pull
docker-compose -f docker/docker-compose.yml build
```

Start the database container:
```bash
docker-compose -f docker/docker-compose.yml up -d cerberus-db
```

Start the simulation:
```bash
docker-compose -f docker/docker-compose.yml run -v ${PWD}/..:/home/sim/project base bash
cd /home/sim_project/flow-simulator
PYTHONPATH=/home/sim/project/flow-simulator/src python3 scripts/cerberus_rotornet_expander_k16_poisson_40G.py
```

For evaluation, install the requirements
```bash
pip3 install -r requirements.txt
```
Afterwards, `evaluation.py` can be run
```bash
python3 evaluation.py
```
It is formated in Scientific|Cell mode as supported by JetBrains Pycharm or Spyder.
