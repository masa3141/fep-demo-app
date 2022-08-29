import streamlit as st
import simple_maze
import readme
import bandit_simulation
import bandit_estimation


def main():
    # Mapping application name to function
    apps = {
        "Bandit Simulation": bandit_simulation.main,
        "Bandit Estimation": bandit_estimation.main,
        # "Readme": readme.main,
        # "simple maze": simple_maze.main,
    }
    selected_app_name = st.sidebar.selectbox(label="Please select a task", options=list(apps.keys()))

    # call the selected function
    render_func = apps[selected_app_name]
    render_func()


if __name__ == "__main__":
    main()
