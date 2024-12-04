# For intro to AI gym, see https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym

# For blackjack and reinforcement learning, see Example 5.1 p. 76 in Sutton & Barto
# In that example, an infinite deck is used

import blackjack_extended as bjk #The extended Black-Jack environment
import blackjack_base as bjk_base #The standard sum-based Black-Jack environment
from math import inf
import RL as rl
import sys
import os
import seaborn as sns
import numpy as np
import pandas as pd

import plotting as pl
import time
import matplotlib 
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

if __name__ == "__main__":
    directory = "{}/data".format(sys.path[0])
    if not os.path.exists(directory):
        os.makedirs(directory)
    path_fun = lambda x: "{}/{}_{}.txt".format(directory,x, decks)
    # init constants
    omega = 0.89    # power decay of the learning rate
    n_sims = 10 ** 5  # Number of episodes generated
    epsilon = 0.05     # Probability in epsilon-soft strategy
    init_val = 0.0
    warmup = n_sims//10
    # Directory to save plots in
    plot_dir = "{}/figures/".format(sys.path[0])


    for decks in [inf]: #1,2,6,8,
        print("----- deck number equal to {} -----".format(decks))
        # set seed
        seed = 31233
        # init envs.
        env = bjk.BlackjackEnvExtend(decks=decks, seed=seed)
        sum_env = bjk_base.BlackjackEnvBase(decks=decks, seed=seed)

        print("----- Starting MC training on expanded state space -----")
        # MC-learning wit expanded state representation
        start_time_MC = time.time()
        Q_MC, MC_avg_reward, state_action_count = rl.learn_MC(
            env, n_sims, gamma = 1, epsilon = epsilon, init_val = init_val,
            episode_file=path_fun("hand_MC_state"), warmup=warmup)
        print("Number of explored states: " + str(len(Q_MC)))
        print("Cumulative avg. reward = " + str(MC_avg_reward))
        time_to_completion_MC = time.time() - start_time_MC

        print("----- Starting Q-learning on expanded state space -----")
        # Q-learning with expanded state representation
        start_time_expanded = time.time()
        Q, avg_reward, state_action_count = rl.learn_Q(
            env, n_sims, gamma = 1, omega = omega, epsilon = epsilon, init_val = init_val,
            episode_file=path_fun("hand_state"), warmup=warmup)
        print("Number of explored states: " + str(len(Q)))
        print("Cumulative avg. reward = " + str(avg_reward))
        time_to_completion_expanded = time.time() - start_time_expanded

        print("----- Starting Q-learning for sum-based state space -----")
            # Q-learning with player sum state representation
        start_time_sum = time.time()
        sumQ, sum_avg_reward, sum_state_action_count = rl.learn_Q(
            sum_env, n_sims, omega = omega, epsilon = epsilon, init_val = init_val,
            episode_file=path_fun("sum_state"), warmup=warmup)
        time_to_completion_sum = time.time() - start_time_sum
        print("Number of explored states (sum states): " + str(len(sumQ)))
        print("Cumulative avg. reward = " + str(sum_avg_reward))

        print("Training time: \n " +
            "Expanded state space MC: {} \n Expanded state space: {} \n Sum state space: {}".format(
                time_to_completion_MC, time_to_completion_expanded, time_to_completion_sum))

    
        # Convert Q (extended state) to sum state representation and make 3D plots
        # Extended state MC-learning
        Q_conv_MC = rl.convert_to_sum_states(Q_MC, env)
        V_conv_MC = rl.convert_to_value_function(Q_conv_MC)
        V_conv_filt_MC = rl.fill_missing_sum_states(rl.filter_states(V_conv_MC))
        pl.plot_value_function(V_conv_filt_MC,
                            title = "Expanded state MC, " + str(decks) + " decks",
                            directory = plot_dir,
                            file_name = "3D_exp_MC_" + str(decks) + "_decks.png")

        # Extended state Q-learning
        Q_conv = rl.convert_to_sum_states(Q, env)
        V_conv = rl.convert_to_value_function(Q_conv)
        V_conv_filt = rl.fill_missing_sum_states(rl.filter_states(V_conv))
        pl.plot_value_function(V_conv_filt,
                            title = "Expanded state, " + str(decks) + " decks",
                            directory = plot_dir,
                            file_name = "3D_exp_" + str(decks) + "_decks_explore.png")
        # Likewise make 3D plots for sumQ
        V_sum = rl.convert_to_value_function(sumQ)
        V_sum_filt = rl.fill_missing_sum_states(rl.filter_states(V_sum))
        pl.plot_value_function(V_sum_filt,
                            title = "Sum state, " + str(decks) + " decks",
                            directory = plot_dir,
                            file_name = "3D_sum_" + str(decks) + "_decks_explore.png")
        # create line plots
        env_types = ["hand_MC", "hand", "sum"]
        fig, lgd = pl.plot_avg_reward_episode(directory, env_types, [str(decks)])
        fig.savefig("{}/avgReturnEp_ndeck{}.png".format(plot_dir, decks),
                                bbox_extra_artists=(lgd,), bbox_inches='tight')
        matplotlib.pyplot.close(fig)


    # create table for optimal strategy (assignment d))
    # Define the range of dealer's showing card and player's sum
    dealer_showing_range = range(2, 12)  # Dealer showing 2 through 11 (Ace)
    player_sum_range = range(12, 22)    # Player sum 12 through 21

    # Create a grid to store the optimal action (0 for "stay", 1 for "hit")
    optimal_strategy_usable = np.zeros((len(player_sum_range), len(dealer_showing_range)))
    optimal_strategy_notusable = np.zeros_like(optimal_strategy_usable)

    usable = True

    for player_sum in player_sum_range:
        for dealer_showing in dealer_showing_range:
            state = (player_sum, dealer_showing, usable)
            optimal_action = 0
            if sumQ[state][1] > sumQ[state][0]:
                optimal_action = 1
            optimal_strategy_usable[player_sum - 12, dealer_showing - 2] = optimal_action
    
    usable = False

    for player_sum in player_sum_range:
        for dealer_showing in dealer_showing_range:
            state = (player_sum, dealer_showing, usable)
            optimal_action = 0
            if sumQ[state][1] > sumQ[state][0]:
                optimal_action = 1
            optimal_strategy_notusable[player_sum-12, dealer_showing-2] = optimal_action

    print(optimal_strategy_usable)

    figure, ax5 = matplotlib.pyplot.subplots(1,1)

    ax5 = sns.heatmap(optimal_strategy_usable, linewidth=0, annot=True, cbar=False, cmap='Accent_r')
    ax5.set_xlabel('Player sum')
    ax5.set_ylabel('Dealer showing')
    ax5.set_xticklabels(player_sum_range)
    ax5.set_yticklabels(['A'] + list(range(2,11)), fontsize = 12)
    
    matplotlib.pyplot.show()

