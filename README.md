# Orchid: An Online Learning based Resource Partitioning Framework for Job Colocation with Multiple Objectives


Orchid is tested on a Ubuntu Server 20.04.4 with Linux 5.4.0 using Python3.8. Please install the following library depencies for running Orchid.
# Dependencies
```
pip3 install numpy  
pip3 install itertools 
apt-get install intel-cmt-cat
```

Orchid uses the Linux _perf_ to collect runtime status of each job for guiding the quick and smart exploration. 
Please ensure that Intel CAT, MBA, and taskset tools are supported and active in your system.
Click this link to confirm: [https://github.com/intel/intel-cmt-cat][check]

# The benchmark suites evaluated in Orchid

PARSEC 3.0: https://parsec.cs.princeton.edu/parsec3-doc.htm

CloudSuite 3.0:https://github.com/parsa-epfl/cloudsuite

ECP Suite 4.0: https://www.exascaleproject.org/


# Run Orchid

## File Description
```
pareto.py : get pareto optimal set.
tools.py : some functions used to execute instructions.
gen_MorpLinUCB.py: main algorithm of Orchid
train.py : main file used to make online resource partitioning decisions
```
## Optional Parameters
```
can set optional parameters in train.py, include:
    :param rounds: The Allowed Sampling Times For a Colocation
    :param alpha_fair: Explore-exploit parameter for objective--fairness
    :param alpha_th: Explore-exploit parameter for objective--throughput
    :param new_alpha_fair: The updated alpha_fair
    :param new_alpha_th:  The updated alpha_th
    :param factor_fair: The factor for update alpha_fair dynamically
    :param factor_th: The factor for update alpha_th dynamically
    :param prob: the probability threshold of choosing arms in a random way in the Pareto Optimal Set
    :param upper_lose_rounds:the required replace times for a bandit version, i.e., the param K in the paper
    :param num_bandit_version: the number of maintained versions of bandits, i.e., the param X in the paper
    :param multi_version_rounds: the time step to start multiple versions mechanism
                                #Note that this mechanism is meaningless in the initialization phase and is added after running MAB for some time steps
```                               
## run program:
    python train.py
