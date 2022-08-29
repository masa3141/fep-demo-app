from collections import defaultdict
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools


def map_estimation(all_ns, all_outcomes, all_selected_indexes, parameters, grids, prior_funcs=None):
    points_list = []
    for i, parameter in enumerate(parameters):
        grid = grids[i]
        start, end, num = grid
        points = np.linspace(start, end, num=num)
        points_list.append(points)
    all_points = list(itertools.product(*points_list))

    log_ls = []
    for point in all_points:
        params = dict()
        log_l = 0
        for i, parameter in enumerate(parameters):
            params[parameter] = point[i]
            if prior_funcs is not None:
                log_l += prior_funcs[i](point[i])

        for t in range(len(all_ns)):
            n = all_ns[t]
            outcomes = all_outcomes[t]
            selected_indexes = all_selected_indexes[t]
            log_l += calc_log_likelihood(n, outcomes, selected_indexes, **params)
        log_ls.append(log_l)
    argmax_index = np.argmax(log_ls)
    estimated_param = all_points[argmax_index]
    result = dict()
    for i, parameter in enumerate(parameters):
        result[parameter] = estimated_param[i]
    return result


def calc_log_likelihood(n, outcomes, selected_indexes, rs=1.0, gamma=1, a0=1, eta=1.0, log_l=0):
    log_l = log_l  # -1.0 / 2 * (rs - 2) ** 2 + -1.0 / 2 * (gamma - 2) ** 2
    n_wins = [0, 0, 0]
    n_looses = [0, 0, 0]
    n_plays = [0, 0, 0]
    rs = rs
    a0 = a0
    eta = eta
    gamma = gamma
    C = np.exp([rs, 0]) / np.sum(np.exp([rs, 0]))

    a1win = a0
    a1loose = a0
    a2win = a0
    a2loose = a0
    a3win = a0
    a3loose = a0

    r1 = a1win / (a1win + a1loose)
    r2 = a2win / (a2win + a2loose)
    r3 = a3win / (a3win + a3loose)

    for t in range(n):

        pragmatic1 = -r1 * np.log(C[0] / C[1]) - np.log(C[1])
        pragmatic2 = -r2 * np.log(C[0] / C[1]) - np.log(C[1])
        pragmatic3 = -r3 * np.log(C[0] / C[1]) - np.log(C[1])

        novelty1 = -1 / (2 * (a1win + a1loose))
        novelty2 = -1 / (2 * (a2win + a2loose))
        novelty3 = -1 / (2 * (a3win + a3loose))

        G = np.array([0.0, 0.0, 0.0])

        G[0] = novelty1 + pragmatic1
        G[1] = novelty2 + pragmatic2
        G[2] = novelty3 + pragmatic3

        p_pi = np.exp(-gamma * G) / np.sum(np.exp(-gamma * G))

        arm_index = selected_indexes[t]

        # add log likelihood
        log_l += np.log(p_pi[arm_index])

        # update A matrix based on a outcome
        outcome = outcomes[t]
        n_plays[arm_index] += 1
        n_wins[arm_index] += outcome
        n_looses[arm_index] += 1 - outcome

        a1win = a0 + eta * n_wins[0]
        a1loose = a0 + eta * n_looses[0]
        a2win = a0 + eta * n_wins[1]
        a2loose = a0 + eta * n_looses[1]
        a3win = a0 + eta * n_wins[2]
        a3loose = a0 + eta * n_looses[2]

        r1 = a1win / (a1win + a1loose)
        r2 = a2win / (a2win + a2loose)
        r3 = a3win / (a3win + a3loose)
    return log_l


def main():
    st.markdown(
        """
        ## Estimate parameters from behavior data

        ### Parameters
        - rs: risk-seeking parameter in C matrix (default=1)
        - 𝛾 : action precision parameter (default=1)
        - a0 : concentration parameters of the beta distribution in a (default=1)
        - 𝜂 : learning rate which controls the magnitude of updates in "A" matrix after each observation (default=1)

        (In this version, rs and 𝛾 can be estimated)


        ### CSV format
        CSV file should contain following columns
        - user_id: unique id for each user
        - trial: number of trials for each user (index from 0)
        - time: the n-th play in one trial (index from 0)
        - selected_arm: selected arm index (index from 0)
        - outcome: if win, 1. if loose, 0
        
        For example, please run a simulation in "Bandit Simulation" page and download the results.
        Then upload the data.
        """
    )
    st.markdown("## Please upload behaviour data")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is None:
        return
    df = pd.read_csv(uploaded_file, sep=",")
    for col in ["user_id", "trial", "time", "selected_arm", "outcome"]:
        if col not in df.columns:
            st.error(f"file should contain {col} column")
    st.dataframe(df)

    # data for parameter estimation
    user_ns = defaultdict(list)
    user_outcomes = defaultdict(list)
    user_selected_indexes = defaultdict(list)
    for (user_id, trial), value in df.groupby(["user_id", "trial"]):
        user_outcomes[user_id].append(value.sort_values("time", ascending=True)["outcome"].tolist())
        user_selected_indexes[user_id].append(value.sort_values("time", ascending=True)["selected_arm"].tolist())
        user_ns[user_id].append(len(value["time"]))

    st.markdown("## Parameters to estimate")
    st.text("Please select parameters to estimate")
    is_rs_estimate = st.checkbox("rs: risk-seeking parameter in C matrix")
    is_gamma_estimate = st.checkbox("𝛾 : action precision parameter")
    estimate_parameters = {"rs": is_rs_estimate, "gamma": is_gamma_estimate}
    st.markdown("## Method to estimate parameters")

    method = st.radio(
        "Please select a method to estimate parameters",
        ("Maximum Likelihood", "Maximum a posteriori(MAP)", "MCMC(Metropolis-Hastings)"),
    )
    if method == "Maximum Likelihood":
        grids = []
        params = []
        for param, is_estimate in estimate_parameters.items():
            if is_estimate:
                st.markdown(f"### Please set a grid search setting for the parameter {param}")
                start = st.number_input(
                    "Start value for grid search",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.0,
                    step=0.1,
                    key=f"grid_start_{param}",
                )
                end = st.number_input(
                    "End value for grid search",
                    min_value=0.0,
                    max_value=10.0,
                    value=5.0,
                    step=0.1,
                    key=f"grid_end_{param}",
                )
                num = st.number_input(
                    "Number of grid",
                    min_value=1,
                    max_value=50,
                    value=10,
                    step=1,
                    key=f"grid_num_{param}",
                )
                grids.append((start, end, num))
                params.append(param)
        button = st.button("Start Estimation")
        if button:
            estimated_results = []
            for user_id, all_outcomes in user_outcomes.items():
                all_selected_indexes = user_selected_indexes[user_id]
                all_ns = user_ns[user_id]
                estimate_result = map_estimation(all_ns, all_outcomes, all_selected_indexes, params, grids)
                estimate_result["user_id"] = user_id
                estimated_results.append(estimate_result)
            pd_estimated_results = pd.DataFrame.from_records(estimated_results, columns=["user_id", *params])
            st.dataframe(pd_estimated_results)

            st.download_button(
                "Download estimation result csv",
                pd_estimated_results.to_csv(index=None).encode("utf-8"),
                "result.csv",
                "text/csv",
                key="estimation-result-csv",
            )
    elif method == "Maximum a posteriori(MAP)":
        grids = []
        params = []
        prior_funcs = []
        for param, is_estimate in estimate_parameters.items():
            if is_estimate:
                st.markdown(f"### Please set Gauss parameters for {param}")
                st.text(f"In this version, Gauss distribution is used as prior distribution for all parameters")

                myu = st.number_input(
                    "mean μ in gauss distribution",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    key=f"gauss_myu_{param}",
                )
                sigma = st.number_input(
                    "standard deviation σ",
                    min_value=0.01,
                    max_value=5.0,
                    value=1.0,
                    step=0.01,
                    key=f"gauss_sigma_{param}",
                )
                st.markdown(f"### Please set a grid search setting for {param}")
                start = st.number_input(
                    "Start value for grid search",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.0,
                    step=0.1,
                    key=f"map_grid_start_{param}",
                )
                end = st.number_input(
                    "End value for grid search",
                    min_value=0.0,
                    max_value=10.0,
                    value=5.0,
                    step=0.1,
                    key=f"map_grid_end_{param}",
                )
                num = st.number_input(
                    "Number of grid",
                    min_value=1,
                    max_value=50,
                    value=10,
                    step=1,
                    key=f"ma_grid_num_{param}",
                )

                prior_func = lambda x: -1.0 / 2 * ((x - myu) / 2) ** 2
                prior_funcs.append(prior_func)

                grids.append((start, end, num))
                params.append(param)
        button = st.button("Start Estimation")
        if button:
            estimated_results = []
            for user_id, all_outcomes in user_outcomes.items():
                all_selected_indexes = user_selected_indexes[user_id]
                all_ns = user_ns[user_id]
                estimate_result = map_estimation(all_ns, all_outcomes, all_selected_indexes, params, grids, prior_funcs)
                estimate_result["user_id"] = user_id
                estimated_results.append(estimate_result)
            pd_estimated_results = pd.DataFrame.from_records(estimated_results, columns=["user_id", *params])
            st.dataframe(pd_estimated_results)

            st.download_button(
                "Download estimation result csv",
                pd_estimated_results.to_csv(index=None).encode("utf-8"),
                "result.csv",
                "text/csv",
                key="estimation-result-csv",
            )

    else:
        st.text("under construction")


if __name__ == "__main__":
    main()
