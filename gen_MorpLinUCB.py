import numpy as np
import random
from numpy.linalg import inv
from pareto import fast_non_dominated_sort

class Morp_LinUCB():
    def __init__(self, ndims, alpha_fair, alpha_th, num_app, core_narms, llc_narms, band_namrms,
                 factor_fair, factor_th, prob):
        self.num_app = num_app
        self.core_narms = core_narms
        self.llc_narms = llc_narms
        self.band_namrms = band_namrms
        self.ndims = ndims
        # Explore-exploit parameters
        self.alpha_fair = alpha_fair
        self.alpha_th = alpha_th
        self.factor_fair = factor_fair
        self.factor_th = factor_th
        # the probability threshold of choosing arms in a random way in the Pareto Optimal Set
        self.prob = prob

        # CPU core bandits: throughput-orient bandit and fairmess-orient bandit
        self.A_c_fair = np.zeros((self.core_narms, self.ndims, self.ndims))
        self.b_c_fair = np.zeros((self.core_narms, self.ndims, 1))
        self.p_c_t_fair = np.zeros(self.core_narms)
        self.A_c_th = np.zeros((self.core_narms, self.ndims, self.ndims))
        self.b_c_th = np.zeros((self.core_narms, self.ndims, 1))
        self.p_c_t_th = np.zeros(self.core_narms)

        # LLC bandits
        self.A_l_fair = np.zeros((self.llc_narms, self.ndims, self.ndims))
        self.b_l_fair = np.zeros((self.llc_narms, self.ndims, 1))
        self.p_l_t_fair = np.zeros(self.llc_narms)
        self.A_l_th = np.zeros((self.llc_narms, self.ndims, self.ndims))
        self.b_l_th = np.zeros((self.llc_narms, self.ndims, 1))
        self.p_l_t_th = np.zeros(self.llc_narms)

        # MB bandits
        self.A_b_fair = np.zeros((self.band_namrms, self.ndims, self.ndims))
        self.b_b_fair = np.zeros((self.band_namrms, self.ndims, 1))
        self.p_b_t_fair = np.zeros(self.band_namrms)
        self.A_b_th = np.zeros((self.band_namrms, self.ndims, self.ndims))
        self.b_b_th = np.zeros((self.band_namrms, self.ndims, 1))
        self.p_b_t_th = np.zeros(self.band_namrms)

        # initialize bandits
        for arm in range(self.core_narms):
            self.A_c_fair[arm] = np.eye(self.ndims)
            self.A_c_th[arm] = np.eye(self.ndims)

        for arm in range(self.llc_narms):
            self.A_l_fair[arm] = np.eye(self.ndims)
            self.A_l_th[arm] = np.eye(self.ndims)

        for arm in range(self.band_namrms):
            self.A_b_fair[arm] = np.eye(self.ndims)
            self.A_b_th[arm] = np.eye(self.ndims)

        super().__init__()
        return

    def update(self, core_arms, llc_arms, band_arms, fair_reward, th_reward, context):
        arm_index = core_arms
        context_core = context[0]; context_core = np.squeeze(context_core, axis=0)
        context_llc = context[1]; context_llc = np.squeeze(context_llc, axis=0)
        context_mb = context[2]; context_mb = np.squeeze(context_mb, axis=0)

        self.A_c_fair[arm_index] += np.outer(np.array(context_core),
                                       np.array(context_core))
        self.b_c_fair[arm_index] = np.add(self.b_c_fair[arm_index].T,
                                    np.array(context_core) * fair_reward).reshape(
            self.ndims, 1)
        self.A_c_th[arm_index] += np.outer(np.array(context_core),
                                             np.array(context_core))
        self.b_c_th[arm_index] = np.add(self.b_c_th[arm_index].T,
                                          np.array(context_core) * th_reward).reshape(
            self.ndims, 1)

        arm_index = llc_arms
        self.A_l_fair[arm_index] += np.outer(np.array(context_llc),
                                       np.array(context_llc))
        self.b_l_fair[arm_index] = np.add(self.b_l_fair[arm_index].T,
                                    np.array(context_llc) * fair_reward).reshape(
            self.ndims, 1)
        self.A_l_th[arm_index] += np.outer(np.array(context_llc),
                                             np.array(context_llc))
        self.b_l_th[arm_index] = np.add(self.b_l_th[arm_index].T,
                                          np.array(context_llc) * th_reward).reshape(
            self.ndims, 1)

        arm_index = band_arms
        self.A_b_fair[arm_index] += np.outer(np.array(context_mb),
                                       np.array(context_mb))
        self.b_b_fair[arm_index] = np.add(self.b_b_fair[arm_index].T,
                                    np.array(context_mb) * fair_reward).reshape(
            self.ndims, 1)
        self.A_b_th[arm_index] += np.outer(np.array(context_mb),
                                             np.array(context_mb))
        self.b_b_th[arm_index] = np.add(self.b_b_th[arm_index].T,
                                          np.array(context_mb) * th_reward).reshape(
            self.ndims, 1)

    def play(self, context):
        context_core = context[0]; context_core = np.squeeze(context_core, axis=0)
        context_llc = context[1]; context_llc = np.squeeze(context_llc, axis=0)
        context_mb = context[2]; context_mb = np.squeeze(context_mb, axis=0)

        # Core bandits
        A_fair = self.A_c_fair
        b_fair = self.b_c_fair
        A_th = self.A_c_th
        b_th = self.b_c_th
        # Update the estimated distributions
        for i in range(self.core_narms):
            theta_fair = inv(A_fair[i]).dot(b_fair[i])
            theta_th = inv(A_th[i]).dot(b_th[i])
            cntx = np.array(context_core)
            self.p_c_t_fair[i] = theta_fair.T.dot(cntx) + self.alpha_fair * np.sqrt(cntx.dot(inv(A_fair[i]).dot(cntx)))
            self.p_c_t_th[i] = theta_th.T.dot(cntx) + self.alpha_th * np.sqrt(cntx.dot(inv(A_th[i]).dot(cntx)))
        core_front = fast_non_dominated_sort(self.p_c_t_fair, self.p_c_t_th)[0]
        core_front_objgap = []
        exp_core_fair_max = self.p_c_t_fair.max()
        exp_core_th_max = self.p_c_t_th.max()
        self.p_c_t_fair = self.p_c_t_fair / exp_core_fair_max
        self.p_c_t_th = self.p_c_t_th / exp_core_th_max
        for i in core_front:
            core_front_objgap.append(abs(self.p_c_t_fair[i] - self.p_c_t_th[i]))
        core_front_objgap = np.array(core_front_objgap)
        if random.random() <= self.prob:
            core_arms_index = np.random.choice(core_front)
        else:
            core_arms_index = core_front[np.random.choice(np.where(core_front_objgap == min(core_front_objgap))[0])]
        core_max = self.p_c_t_fair[core_arms_index] * self.p_c_t_th[core_arms_index]


        # LLC bandits
        A_fair = self.A_l_fair
        b_fair = self.b_l_fair
        A_th = self.A_l_th
        b_th = self.b_l_th
        for i in range(self.llc_narms):
            theta_fair = inv(A_fair[i]).dot(b_fair[i])
            theta_th = inv(A_th[i]).dot(b_th[i])
            cntx = np.array(context_llc)
            self.p_l_t_fair[i] = theta_fair.T.dot(cntx) + self.alpha_fair * np.sqrt(cntx.dot(inv(A_fair[i]).dot(cntx)))
            self.p_l_t_th[i] = theta_th.T.dot(cntx) + self.alpha_th * np.sqrt(cntx.dot(inv(A_th[i]).dot(cntx)))
        llc_front = fast_non_dominated_sort(self.p_l_t_fair, self.p_l_t_th)[0]
        llc_front_objgap = []
        exp_llc_fair_max = self.p_l_t_fair.max()
        exp_llc_th_max = self.p_l_t_th.max()
        self.p_l_t_fair = self.p_l_t_fair / exp_llc_fair_max
        self.p_l_t_th = self.p_l_t_th / exp_llc_th_max
        for i in llc_front:
            llc_front_objgap.append(abs(self.p_l_t_fair[i] - self.p_l_t_th[i]))
        llc_front_objgap = np.array(llc_front_objgap)
        if random.random() <= self.prob:
            llc_arms_index = np.random.choice(llc_front)
        else:
            llc_arms_index = llc_front[np.random.choice(np.where(llc_front_objgap == min(llc_front_objgap))[0])]
        llc_max = self.p_l_t_fair[llc_arms_index] * self.p_l_t_th[llc_arms_index]


        # MB bandits
        A_fair = self.A_b_fair
        b_fair = self.b_b_fair
        A_th = self.A_b_th
        b_th = self.b_b_th
        for i in range(self.band_namrms):
            theta_fair = inv(A_fair[i]).dot(b_fair[i])
            theta_th = inv(A_th[i]).dot(b_th[i])
            cntx = np.array(context_mb)
            self.p_b_t_fair[i] = theta_fair.T.dot(cntx) + self.alpha_fair * np.sqrt(cntx.dot(inv(A_fair[i]).dot(cntx)))
            self.p_b_t_th[i] = theta_th.T.dot(cntx) + self.alpha_th * np.sqrt(cntx.dot(inv(A_th[i]).dot(cntx)))
        band_front = fast_non_dominated_sort(self.p_b_t_fair, self.p_b_t_th)[0]
        band_front_objgap = []
        exp_band_fair_max = self.p_b_t_fair.max()
        exp_band_th_max = self.p_b_t_th.max()
        self.p_b_t_fair = self.p_b_t_fair / exp_band_fair_max
        self.p_b_t_th = self.p_b_t_th / exp_band_th_max
        for i in band_front:
            band_front_objgap.append(abs(self.p_b_t_fair[i] - self.p_b_t_th[i]))
        band_front_objgap = np.array(band_front_objgap)
        if random.random() <= self.prob:
            band_arms_index = np.random.choice(band_front)
        else:
            band_arms_index = band_front[np.random.choice(np.where(band_front_objgap == min(band_front_objgap))[0])]
        band_max = self.p_b_t_fair[band_arms_index] * self.p_b_t_th[band_arms_index]

        # Update explore - exploit parameters dynamically
        self.alpha_fair *= self.factor_fair
        self.alpha_th *= self.factor_th
        return core_arms_index, llc_arms_index, band_arms_index, core_max, llc_max, band_max