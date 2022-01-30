import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import streamlit as st
from utils import *
from pymdp import utils
from pymdp.maths import softmax
from pymdp.maths import spm_log_single as log_stable
from pymdp.control import construct_policies


grid_locations = list(itertools.product(range(3), repeat=2))
actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


class GridWorldEnv:
    def __init__(self, starting_state=(0, 0)):

        self.init_state = starting_state
        self.current_state = self.init_state
        print(f"Starting state is {starting_state}")

    def step(self, action_label):

        (Y, X) = self.current_state

        if action_label == "UP":

            Y_new = Y - 1 if Y > 0 else Y
            X_new = X

        elif action_label == "DOWN":

            Y_new = Y + 1 if Y < 2 else Y
            X_new = X

        elif action_label == "LEFT":
            Y_new = Y
            X_new = X - 1 if X > 0 else X

        elif action_label == "RIGHT":
            Y_new = Y
            X_new = X + 1 if X < 2 else X

        elif action_label == "STAY":
            Y_new, X_new = Y, X

        self.current_state = (Y_new, X_new)  # store the new grid location

        obs = self.current_state  # agent always directly observes the grid location they're in

        return obs

    def reset(self):
        self.current_state = self.init_state
        print(f"Re-initialized location to {self.init_state}")
        obs = self.current_state
        print(f"..and sampled observation {obs}")

        return obs


def run_active_inference_loop(A, B, C, D, actions, env, T=5):

    """Initialize the prior that will be passed in during inference to be the same as `D`"""
    prior = D.copy()  # initial prior should be the D vector

    """ Initialize the observation that will be passed in during inference - hint use env.reset()"""
    obs = (
        env.reset()
    )  # initialize the `obs` variable to be the first observation you sample from the environment, before `step`-ing it.

    for t in range(T):

        print(f"Time {t}: Agent observes itself in location: {obs}")

        # convert the observation into the agent's observational state space (in terms of 0 through 8)
        obs_idx = grid_locations.index(obs)

        # perform inference over hidden states
        qs_current = infer_states(obs_idx, A, prior)

        plot_beliefs(qs_current, title_str=f"Beliefs about location at time {t}")
        # plot_grid_with_probability(grid_locations, qs_current)

        # calculate expected free energy of actions
        G = calculate_G(A, B, C, qs_current, actions)

        # compute action posterior
        Q_u = softmax(-G)

        # sample action from probability distribution over actions
        chosen_action = utils.sample(Q_u)
        print(f"chosen_action: {chosen_action}")

        # compute prior for next timestep of inference
        prior = B[:, :, chosen_action].dot(qs_current)

        # update generative process
        action_label = actions[chosen_action]

        obs = env.step(action_label)

    return qs_current


def calculate_G_policies(A, B, C, qs_current, policies):

    G = np.zeros(len(policies))  # initialize the vector of expected free energies, one per policy
    H_A = entropy(A)  # can calculate the entropy of the A matrix beforehand, since it'll be the same for all policies

    for policy_id, policy in enumerate(
        policies
    ):  # loop over policies - policy_id will be the linear index of the policy (0, 1, 2, ...) and `policy` will be a column vector where `policy[t,0]` indexes the action entailed by that policy at time `t`

        t_horizon = policy.shape[0]  # temporal depth of the policy

        G_pi = 0.0  # initialize expected free energy for this policy

        for t in range(t_horizon):  # loop over temporal depth of the policy

            action = policy[t, 0]  # action entailed by this particular policy, at time `t`

            # get the past predictive posterior - which is either your current posterior at the current time (not the policy time) or the predictive posterior entailed by this policy, one timstep ago (in policy time)
            if t == 0:
                qs_prev = qs_current
            else:
                qs_prev = qs_pi_t

            qs_pi_t = get_expected_states(
                B, qs_prev, action
            )  # expected states, under the action entailed by the policy at this particular time
            qo_pi_t = get_expected_observations(
                A, qs_pi_t
            )  # expected observations, under the action entailed by the policy at this particular time

            kld = kl_divergence(
                qo_pi_t, C
            )  # Kullback-Leibler divergence between expected observations and the prior preferences C

            G_pi_t = H_A.dot(qs_pi_t) + kld  # predicted uncertainty + predicted divergence, for this policy & timepoint

            G_pi += G_pi_t  # accumulate the expected free energy for each timepoint into the overall EFE for the policy

        G[policy_id] += G_pi

    return G


def compute_prob_actions(actions, policies, Q_pi):
    P_u = np.zeros(len(actions))  # initialize the vector of probabilities of each action

    for policy_id, policy in enumerate(policies):
        P_u[int(policy[0, 0])] += Q_pi[
            policy_id
        ]  # get the marginal probability for the given action, entailed by this policy at the first timestep

    P_u = utils.norm_dist(P_u)  # normalize the action probabilities

    return P_u


def active_inference_with_planning(A, B, C, D, n_actions, env, noise_b, policy_len=2, T=5):

    """Initialize prior, first observation, and policies"""

    prior = D  # initial prior should be the D vector

    obs = env.reset()  # get the initial observation

    policies = construct_policies([len(D)], [n_actions], policy_len=policy_len)

    for t in range(T):

        print(f"Time {t}: Agent observes itself in location: {obs}")

        # convert the observation into the agent's observational state space (in terms of 0 through 8)
        obs_idx = grid_locations.index(obs)

        # perform inference over hidden states
        qs_current = infer_states(obs_idx, A, prior)
        # plot_beliefs(qs_current, title_str=f"Beliefs about location at time {t}")
        print(f"qs_current={qs_current}")
        plot_grid_with_probability(grid_locations, qs_current, desc=f"Beliefs about location at time {t}")

        # calculate expected free energy of actions
        G = calculate_G_policies(A, B, C, qs_current, policies)
        for i, g in enumerate(G[0:3]):
            st.write(f"Expected free energy of policy{i} ={g:0.1f} at time {t}")
        st.write("......")
        # to get action posterior, we marginalize P(u|pi) with the probabilities of each policy Q(pi), given by \sigma(-G)
        Q_pi = softmax(-G)

        # compute the probability of each action
        P_u = compute_prob_actions(actions, policies, Q_pi)
        for i, p_u in enumerate(P_u):
            st.write(f"probability of action {actions[i]} ={p_u} at time {t}")

        # sample action from probability distribution over actions
        chosen_action = utils.sample(P_u)
        st.write(f"Chosen action at time {t} is {actions[chosen_action]} ")

        # compute prior for next timestep of inference
        prior = B[:, :, chosen_action].dot(qs_current) + noise_b
        prior = prior / np.sum(prior)

        # step the generative process and get new observation
        action_label = actions[chosen_action]
        obs = env.step(action_label)

    return qs_current


def calculate_G(A, B, C, qs_current, actions):

    G = np.zeros(len(actions))  # vector of expected free energies, one per action

    H_A = entropy(A)  # entropy of the observation model, P(o|s)

    for action_i in range(len(actions)):

        qs_u = get_expected_states(
            B, qs_current, action_i
        )  # expected states, under the action we're currently looping over
        qo_u = get_expected_observations(
            A, qs_u
        )  # expected observations, under the action we're currently looping over

        pred_uncertainty = H_A.dot(qs_u)  # predicted uncertainty, i.e. expected entropy of the A matrix
        pred_div = kl_divergence(qo_u, C)  # predicted divergence

        G[action_i] = pred_uncertainty + pred_div  # sum them together to get expected free energy

    return G


def create_B_matrix():
    B = np.zeros((len(grid_locations), len(grid_locations), len(actions)))

    for action_id, action_label in enumerate(actions):

        for curr_state, grid_location in enumerate(grid_locations):

            y, x = grid_location

            if action_label == "UP":
                next_y = y - 1 if y > 0 else y
                next_x = x
            elif action_label == "DOWN":
                next_y = y + 1 if y < 2 else y
                next_x = x
            elif action_label == "LEFT":
                next_x = x - 1 if x > 0 else x
                next_y = y
            elif action_label == "RIGHT":
                next_x = x + 1 if x < 2 else x
                next_y = y
            elif action_label == "STAY":
                next_x = x
                next_y = y
            new_location = (next_y, next_x)
            next_state = grid_locations.index(new_location)
            B[next_state, curr_state, action_id] = 1.0
    return B


def infer_states(observation_index, A, prior):
    print("infer_states")

    """Implement inference here -- NOTE: prior is already passed in, so you don't need to do anything with the B matrix."""
    """ This function has already been given P(s_t). The conditional expectation that creates "today's prior", using "yesterday's posterior", will happen *before calling* this function"""

    log_likelihood = log_stable(A[observation_index, :])

    log_prior = log_stable(prior)

    qs = softmax(log_likelihood + log_prior)

    return qs


""" define component functions for computing expected free energy """


def get_expected_states(B, qs_current, action):
    """Compute the expected states one step into the future, given a particular action"""
    qs_u = B[:, :, action].dot(qs_current)

    return qs_u


def get_expected_observations(A, qs_u):
    """Compute the expected observations one step into the future, given a particular action"""

    qo_u = A.dot(qs_u)

    return qo_u


def entropy(A):
    """Compute the entropy of a set of conditional distributions, i.e. one entropy value per column"""

    H_A = -(A * log_stable(A)).sum(axis=0)

    return H_A


def kl_divergence(qo_u, C):
    """Compute the Kullback-Leibler divergence between two 1-D categorical distributions"""

    return (log_stable(qo_u) - log_stable(C)).dot(qo_u)


def main():
    st.header("Simple maze")
    st.markdown(
        """
        This is based on the [Tutorial 1: Active inference from scratch](https://pymdp-rtd.readthedocs.io/en/latest/notebooks/active_inference_from_scratch.html) \n
        This interactive demo app shows how to compute the expected free energy in 3*3 maze.
        """
    )

    st.markdown("## 3*3 maze")

    plot_grid(grid_locations)

    st.markdown("## Building the generative model: A, B, C and D")
    st.markdown("### The A matrix P(o|s)")
    st.markdown("#### ")
    n_states = len(grid_locations)
    n_observations = len(grid_locations)
    st.markdown(f"- Dimensionality of hidden states: {n_states}")
    st.markdown(f"- Dimensionality of observations: {n_observations}")
    st.markdown(
        """
    #### Please choose a noise intensity.\n 
    A = A + noise_parameter x random(A.shape) \n
    A = normalize(A)
    """
    )

    a_noize_parameter = st.slider("noise_parameter of A", 0.0, 1.0, value=0.0, step=0.01)
    A = np.zeros((n_states, n_observations))
    np.fill_diagonal(A, 1.0)
    A = a_noize_parameter * np.random.random(A.shape) + A
    print(A)
    A = utils.norm_dist(A)
    print(A)
    plot_likelihood(A, title_str="A matrix or $P(o|s)$")

    st.markdown("### The B matrix P(s_t|s_t-1, u_t-1)")
    st.write(
        "Prior beliefs about transitions between hidden states over time. These transitions are conditioned on previous hidden state s(t-1) and past action u(t-1)"
    )
    st.markdown('actions: ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]')
    st.write("Size of B matrix is (len(grid_locations), len(grid_locations), len(actions)) = (9, 9, 5)")
    B = create_B_matrix()

    starting_location = (1, 0)
    state_index = grid_locations.index(starting_location)
    starting_state = utils.onehot(state_index, n_states)

    st.markdown(
        """
    ### The prior over observations: the C vector P(o)
    Setting desired rewarding index to (2,2), correspondig to index 8
    """
    )

    """ Create an empty vector to store the preferences over observations """
    C = np.zeros(n_observations)
    """ Choose an observation index to be the 'desired' rewarding index, and fill out the C vector accordingly """
    desired_location = (2, 2)  # choose a desired location
    desired_location_index = grid_locations.index(
        desired_location
    )  # get the linear index of the grid location, in terms of 0 through 8
    C[desired_location_index] = 1.0  # set the preference for that location to be 100%, i.e. 1.0
    """  Let's look at the prior preference distribution """
    plot_beliefs(C, title_str="Preferences over observations")

    st.markdown(
        """
    ### The prior over observations: the D vector P(s)
    Prior belief over hidden states at the first timestep.
    """
    )
    """ Create a D vector, basically a belief that the agent has about its own starting location """

    # create a one-hot / certain belief about initial state
    D = utils.onehot(0, n_states)

    # d_noize_parameter = st.slider("noise_parameter of D", 0.0, 1.0, value=0.0, step=0.01)
    # D = d_noize_parameter * np.random.random(D.shape) + D
    # D = D / np.sum(D)
    """ Let's look at the prior over hidden states """
    plot_beliefs(D, title_str="Prior beliefs over states")

    st.markdown("## Hidden state inference")
    st.latex(
        r"""
    q(s_t) = \sigma\left(\ln \mathbf{A}[o,:] + \ln\mathbf{B}[:,:,u] \cdot q(s_{t-1})\right) 
    """
    )
    st.latex(
        r"""
P(s_t) = \mathbf{E}_{q(s_{t-1})}\left[P(s_t | s_{t-1}, u_{t-1})\right]
    """
    )
    # qs_past = utils.onehot(4, n_states)  # agent believes they were at location 4 -- i.e. (1,1) one timestep ago

    # last_action = "UP"  # the agent knew it moved "UP" one timestep ago
    # action_id = actions.index(last_action)  # get the action index for moving "UP"
    # prior = B[:, :, action_id].dot(qs_past)

    # observation_index = 1
    # qs_new = infer_states(observation_index, A, prior)
    # print("A", A)
    # print("prior", prior)
    # print("log_stable(prior)", log_stable(prior))
    # print("log_stable(A[observation_index,:])", log_stable(A[observation_index, :]))
    # print("qs_new", qs_new)
    # print("A sum", A.sum(axis=1))
    # print("A sum=0", A.sum(axis=0))
    # plot_beliefs(qs_new, title_str="Beliefs about hidden states")
    st.markdown(
        """
    ## Complete Recipe for Active Inference
1. Sample an observation 𝑜𝑡 from the current state of the environment
2. Perform inference over hidden states i.e., optimize 𝑞(𝑠) through free-energy minimization
3. Calculate expected free energy of actions 𝐆
4. Sample action from the posterior over actions 𝑄(𝑢𝑡)∼𝜎(−𝐆).
5. Use the sampled action 𝑎𝑡 to perturb the generative process and go back to step 1.
    """
    )
    n_actions = len(actions)

    D = utils.onehot(
        grid_locations.index((0, 0)), n_states
    )  # let's have the agent believe it starts in location (0,0) (upper left corner)
    env = GridWorldEnv(starting_state=(0, 0))

    st.markdown("## Simulation")
    policy_len = st.number_input("Choose a planning horizon", min_value=1, max_value=5, value=3, step=1)
    noise_b = np.random.random(len(D)) * st.slider("Chose noise_parameter of B", 0.0, 1.0, value=0.0, step=0.01)

    st.markdown("### Tips")
    st.markdown("- If a planning horizon=1 and noise=0, the agent can't reach the goal")
    st.markdown("- If a planning horizon=3, the agent can reach the goal")
    st.write("Please click the Start button to start simulation")
    increment = st.button("Start")
    if increment:
        qs_final = active_inference_with_planning(A, B, C, D, n_actions, env, noise_b, policy_len=policy_len, T=10)


if __name__ == "__main__":
    main()
