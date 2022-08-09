# Orchid: An Online Learning based Resource Partitioning Framework for Job Colocation with Multiple Objectives


Orchid is tested on a Ubuntu Server 20.04.4 with Linux 5.4.0 using Python3.8. Please install the following library depencies for running SATORI.
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
