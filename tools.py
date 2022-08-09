import sys
import numpy as np
import os
import subprocess

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)



def get_now_cntx_reward(core_list, llc_config, mb_config):
    """
    :param core_list, llc_config, mb_config: the current resource configration
    :return: the performance of two objectives
    """
    # Use Perf tool to get the values of all jobs' selected PMCs
    features, fair_reward, th_reward = get_PMCs_reward_from_realtime()
    # Caculate the mean and standard deviation of features according the rules described in the paper
    context = cacul(features)
    return context, fair_reward, th_reward


# Transform the configuration of CPU cores to taskset format, such as [2,4,3] => ["0,1","2,3,4,5","6,7,8"]
def refer_core(core_config):
    app_cores = [""] * len(core_config)
    endpoint_left = 0
    for i in range(len(core_config)):
        endpoint_right = endpoint_left + core_config[i] - 1
        app_cores[i] = ",".join([str(c) for c in list(range(endpoint_left, endpoint_right+1))])
        endpoint_left = endpoint_right + 1
    return app_cores

# Transform the configuration of LLC to CAT format
def refer_llc(llc_config):
    nof_llc = np.array(llc_config).sum()
    i = nof_llc - 1
    llc_list = []
    for j in range(len(llc_config)):
        ini_list = [0 for k in range(nof_llc)]
        count = llc_config[j]
        while count > 0:
            ini_list[i] = 1
            i -= 1
            count -= 1
        llc_list.append(hex(int(''.join([str(item) for item in ini_list]), 2)))
    return llc_list

# Initial configuration, i.e., equally partitioning
def gen_init_config(app_id, core_arm_orders, llc_arm_orders, mb_arm_orders, NUM_UNITS):
    app_num = len(app_id)
    nof_core = NUM_UNITS[0]
    nof_llc = NUM_UNITS[1]
    nof_mb = NUM_UNITS[2]
    # Core configuration initialization
    each_core_config = nof_core // app_num
    res_core_config = nof_core % app_num
    core_config = [each_core_config] * (app_num-1)
    if res_core_config >= each_core_config:
        for i in range(res_core_config):
            core_config[i]+=1
        core_config.append(1)
    else:
        core_config.append(each_core_config+res_core_config)
    for i in range(len(core_arm_orders)):
        if core_arm_orders[i] == core_config:
            core_arms = i
            break
    # LLC configuration initialization
    each_llc_config = nof_llc // app_num
    res_llc_config = nof_llc % app_num
    llc_config = [each_llc_config] * (app_num - 1)
    if res_llc_config >= each_llc_config:
        for i in range(res_llc_config):
            llc_config[i] += 1
        llc_config.append(each_llc_config)
    else:
        llc_config.append(each_llc_config + res_llc_config)
    for i in range(len(llc_arm_orders)):
        if llc_arm_orders[i] == llc_config:
            llc_arms = i
            break
    # MB configuration initialization
    each_mb_config = nof_mb // app_num
    res_mb_config = nof_mb % app_num
    mb_config = [each_mb_config] * (app_num - 1)
    if res_mb_config >= each_mb_config:
        for i in range(res_mb_config):
            mb_config[i] += 1
        mb_config.append(each_mb_config)
    else:
        mb_config.append(each_mb_config + res_mb_config)
    for i in range(len(mb_arm_orders)):
        if mb_arm_orders[i] == mb_config:
            mb_arms = i
            break
    chosen_arms = [core_arms, llc_arms, mb_arms]
    core_list = refer_core(core_config)
    llc_list = refer_llc(llc_config)

    # PID_L is the list of the real PID of the corresponding job
    PID_L = get_pid_from_job_name()

    for i in range(len(core_config)):
        subprocess.call(f'sudo taskset -apc {core_list[i]} {PID_L} > /dev/null', shell=True)
        subprocess.run('sudo pqos -a "llc:{}={}"'.format(i+1, core_list[i]), shell=True, capture_output=True)
        subprocess.run('sudo pqos -e "llc:{}={}"'.format(i+1, llc_list[i]), shell=True, capture_output=True)
        subprocess.run('sudo pqos -a "core:{}={}"'.format(i+1, core_list[i]), shell=True, capture_output=True)
        subprocess.run('sudo pqos -e "mba:{}={}"'.format(i+1, int(float(mb_config[i])) * 10), shell=True,
                       capture_output=True)

    return core_list,llc_config,mb_config,chosen_arms


# Make the configuration
def gen_config(app_id, chosen_arms, core_arm_orders, llc_arm_orders, mb_arm_orders):
    # PID_L is the list of the real PID of the corresponding job
    PID_L = get_pid_from_job_name()
    core_arm, llc_arm, mb_arm = chosen_arms[0], chosen_arms[1], chosen_arms[2]

    core_config = core_arm_orders[core_arm]
    llc_config = llc_arm_orders[llc_arm]
    mb_config = mb_arm_orders[mb_arm]

    core_list = refer_core(core_config)
    llc_list = refer_llc(llc_config)

    for i in range(len(core_config)):
        subprocess.call(f'sudo taskset -apc {core_list[i]} {PID_L[i]} > /dev/null', shell=True)
        subprocess.run('sudo pqos -a "llc:{}={}"'.format(i+1, core_list[i]), shell=True, capture_output=True)
        subprocess.run('sudo pqos -e "llc:{}={}"'.format(i+1, llc_list[i]), shell=True, capture_output=True)
        subprocess.run('sudo pqos -a "core:{}={}"'.format(i+1, core_list[i]), shell=True, capture_output=True)
        subprocess.run('sudo pqos -e "mba:{}={}"'.format(i+1, int(float(mb_config[i])) * 10), shell=True, capture_output=True)
    return core_list, llc_config, mb_config

def gen_configs_recursively(u, r, a, NUM_APPS, NUM_UNITS):
    if (a == NUM_APPS-1):
        return None
    else:
        ret = []
        for i in range(1, NUM_UNITS[r]-u+1-NUM_APPS+a+1):
            confs = gen_configs_recursively(u+i, r, a+1, NUM_APPS, NUM_UNITS)
            if not confs:
                ret.append([i])
            else:
                for c in confs:
                    ret.append([i])
                    for j in c:
                        ret[-1].append(j)
        return ret

# Enumerate each resource's configurations
def get_all_config(NUM_APPS, NUM_UNITS):
    core_config = gen_configs_recursively(0, 0, 0, NUM_APPS, NUM_UNITS)
    for i in range(len(core_config)):
        other_source = np.array(core_config[i]).sum()
        core_config[i].append(NUM_UNITS[0] - other_source)

    llc_config = gen_configs_recursively(0, 1, 0, NUM_APPS, NUM_UNITS)
    for i in range(len(llc_config)):
        other_source = np.array(llc_config[i]).sum()
        llc_config[i].append(NUM_UNITS[1] - other_source)

    mb_config = gen_configs_recursively(0, 2, 0, NUM_APPS, NUM_UNITS)
    for i in range(len(mb_config)):
        other_source = np.array(mb_config[i]).sum()
        mb_config[i].append(NUM_UNITS[2] - other_source)
    return core_config, llc_config, mb_config