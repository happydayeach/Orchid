import random
import sys
import os
import subprocess
import time
import numpy as np
from numpy.linalg import inv
import itertools
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from tools import gen_config, gen_init_config, gen_nonOverlap_config,get_now_cntx_reward
from gen_MorpLinUCB import Morp_LinUCB



def Orchid(rounds, alpha_fair, alpha_th, new_alpha_fair, new_alpha_th,
          factor_fair, factor_th, prob, upper_lose_rounds, num_bandit_version,
          multi_version_rounds):

    """
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
    :return:
    """

    # The number of selected PMCs for two objectives
    nof_counters = 18
    nof_colocation = len(colocation_list)

    # Create MAB
    mab = Morp_LinUCB(nof_counters, alpha_fair=alpha_fair, alpha_th=alpha_th, num_app=len(colocation_list[0]),
                        core_narms=len(core_arm_orders), llc_narms=len(llc_arm_orders),
                        band_namrms=len(mb_arm_orders), factor_fair=factor_fair, factor_th=factor_th, prob=prob)

    for col_items in range(nof_colocation):
        is_make_multi_version = False
        # The arrived colocated jobs
        app_id = colocation_list[col_items]
        reward = []
        mab_new = []
        mab_new_chosen_arms = []
        mab_new_p = []
        lose_rounds = 0
        best_mab_round = 0
        N_bandit_chosen = np.zeros((num_bandit_version + 1, len(NUM_UNITS)))

        core_list, llc_config, mb_config, chosen_arms = gen_init_config(app_id, core_arm_orders,
                                                                                    llc_arm_orders,
                                                                                    mb_arm_orders,
                                                                                    NUM_UNITS)
        for now_round in range(rounds):

            print(f"Start run {now_round+1}th round")
            if now_round == 0:
                context, fair_reward, th_reward = get_now_cntx_reward(core_list, llc_config, mb_config)
                reward.append([fair_reward, th_reward])
                chosen_arms, _ = onlineEvaluate(mab, fair_reward, th_reward, chosen_arms, context, is_update=True)

            else:
                # Get resources configuration
                core_list, llc_config, mb_config = gen_config(app_id, chosen_arms, core_arm_orders, llc_arm_orders,
                                                              mb_arm_orders)
                # Get rewards (i.e., the performance of two objectives)
                context, fair_reward, th_reward = get_now_cntx_reward(core_list, llc_config, mb_config)
                reward.append([fair_reward, th_reward])

                if now_round > multi_version_rounds and is_make_multi_version == False:
                    is_make_multi_version = True
                    for mab_id in range(num_bandit_version):
                        mab_new[mab_id] = Morp_LinUCB(nof_counters, alpha_fair=new_alpha_fair, alpha_th=new_alpha_th,
                                num_app=len(colocation_list[0]), core_narms=len(core_arm_orders), llc_narms=len(llc_arm_orders),
                                band_namrms=len(mb_arm_orders), factor_fair=factor_fair, factor_th=factor_th, prob=prob)
                    print('********** make multi-version mabs ************"')

                if now_round <= multi_version_rounds:
                    chosen_arms, _ = onlineEvaluate(mab, fair_reward, th_reward, chosen_arms, context, is_update=True)

                else:
                    if best_mab_round == 0:
                        mab_chosen_arms, mab_p = onlineEvaluate(mab, fair_reward, th_reward, chosen_arms, context,
                                                                is_update=True)
                    else:
                        mab_chosen_arms, mab_p = onlineEvaluate(mab, fair_reward, th_reward, chosen_arms, context,
                                                                is_update=False)
                    for mab_id in range(num_bandit_version):
                        if best_mab_round == mab_id + 1:
                            mab_new_chosen_arms[mab_id], mab_new_p[mab_id] = onlineEvaluate(mab_new[mab_id], fair_reward,
                                                                                            th_reward, chosen_arms, context,
                                                                                            is_update=True)
                        else:
                            mab_new_chosen_arms[mab_id], mab_new_p[mab_id] = onlineEvaluate(mab_new[mab_id],
                                                                                            fair_reward,
                                                                                            th_reward, chosen_arms,
                                                                                            context,
                                                                                            is_update=False)
                    num_nonoptimal_bandit = 0
                    score_mab_round = [0] * (num_bandit_version + 1)
                    for i in range(3):
                        optimal_bandit = 0
                        optimal = mab_p[i]
                        flag = 0
                        for mab_id in range(num_bandit_version):
                            if optimal <= mab_new_p[mab_id][i]:
                                optimal = mab_new_p[mab_id][i]
                                optimal_bandit = mab_id + 1
                                flag = 1
                        if flag != 0:
                            num_nonoptimal_bandit += 1
                        score_mab_round[optimal_bandit] += 1
                        N_bandit_chosen[optimal_bandit, i] += 1

                    if num_nonoptimal_bandit >= 2:
                        lose_rounds += 1
                        best_mab_round = np.array(score_mab_round).argmax()
                        chosen_arms = mab_new_chosen_arms[best_mab_round-1]
                    else:
                        lose_rounds = 0
                        chosen_arms = mab_chosen_arms
                        best_mab_round = 0

                    if lose_rounds >= upper_lose_rounds:
                        del(mab)
                        optimal_mab = 0
                        optimal_sum = N_bandit_chosen.sum(axis=0)[optimal_mab + 1]
                        for mab_id in range(num_bandit_version):
                            if optimal_sum < N_bandit_chosen.sum(axis=0)[mab_id + 1]:
                                optimal_sum = N_bandit_chosen.sum(axis=0)[mab_id + 1]
                                optimal_mab = mab_id
                        mab = mab_new[optimal_mab]
                        print('*******choose a new mab***************')
                        lose_rounds = 0
                        N_bandit_chosen = np.zeros((num_bandit_version + 1, len(NUM_UNITS)))
                        mab_new[optimal_mab] = Morp_LinUCB(nof_counters, alpha_fair=new_alpha_fair, alpha_th=new_alpha_th,
                                    num_app=len(colocation_list[0]), core_narms=len(core_arm_orders), llc_narms=len(llc_arm_orders),
                                    band_namrms=len(mb_arm_orders), factor_fair=factor_fair, factor_th=factor_th, prob=prob)


            print(f"############{now_round+1}th,{[fair_reward, th_reward]}##################")



def onlineEvaluate(mab, fair_reward, th_reward, chosen_arms, context, is_update):
    # Update the MAB according to the real-time performance of two objectives
    if is_update:
        mab.update(chosen_arms[0], chosen_arms[1], chosen_arms[2], fair_reward=fair_reward, th_reward=th_reward, context=context)
    # The resource configuration is determined according to the probability distributions learned by the MAB so far
    core_action, llc_action, band_action, core_max, llc_max, band_max = mab.play(context)
    chosen_arms = [core_action, llc_action, band_action]
    p_max = [core_max, llc_max, band_max]

    return chosen_arms, p_max

if __name__ == "__main__":
    subprocess.run('sudo pqos -R', shell=True, capture_output=True)
    time.sleep(5)

    # Input the name of PMCs we choose
    evne = [PMCs]

    # The colocation size
    NUM_APPS = 4
    # Max units of (cores, LLC ways, memory bandwidth)
    NUM_UNITS = [10, 10, 10]

    # The name of the colocations coming to the server, like:
    colocation_list = ['canneal', 'fluidanimate', 'freqmine', 'streamcluster']
    # or:
    # colocation_list = [['canneal', 'fluidanimate', 'freqmine', 'blackscholes'], ['streamcluster', 'blackscholes', 'swaptions', 'vips']]


    # Enumerate each resource's configurations
    core_arm_orders, llc_arm_orders, mb_arm_orders = gen_nonOverlap_config(NUM_UNITS=NUM_UNITS, NUM_APPS=NUM_APPS)
    # Set paremeters for Orchid
    rounds = 100
    alpha_fair = 0.08
    alpha_th = 0.018
    new_alpha_fair = 0.04
    new_alpha_th = 0.009
    factor_fair = 0.98     # attenuation factor of alpha_fair, suggest 0.98, a tip which adjusts alpha dynamically
    factor_th = 0.98
    prob = 0.3
    upper_lose_rounds = 15
    num_bandit_version = 3
    multi_version_rounds = 40
    # Start Online Decisioning
    Orchid(rounds=rounds, alpha_fair=alpha_fair, alpha_th=alpha_th, factor_fair=factor_fair,
          factor_th=factor_th, prob=prob, upper_lose_rounds=upper_lose_rounds,
          num_bandit_version=num_bandit_version-1, multi_version_rounds=multi_version_rounds)
