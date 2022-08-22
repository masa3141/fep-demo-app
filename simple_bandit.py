import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


class FEP:
    def __init__(self, rs, gamma, a0, eta, true_prob, n):
        self.rs = rs
        self.gamma = gamma
        self.a0 = a0
        self.eta = eta
        self.true_prob = true_prob
        self.n = n

        self.a1win = a0
        self.a1loose = a0
        self.a2win = a0
        self.a2loose = a0
        self.a3win = a0
        self.a3loose = a0

        self.r1 = self.a1win / (self.a1win + self.a1loose)
        self.r2 = self.a2win / (self.a2win + self.a2loose)
        self.r3 = self.a3win / (self.a3win + self.a3loose)

        self.C = np.exp([self.rs, 0]) / np.sum(np.exp([self.rs, 0]))

        # count of winning/loose/play in each earm
        self.n_wins = [0, 0, 0]
        self.n_looses = [0, 0, 0]
        self.n_plays = [0, 0, 0]

        # G of each arm in each play
        self.G1s = []
        self.G2s = []
        self.G3s = []

        # Actiopn probability of each arm in each play
        self.p1s = []
        self.p2s = []
        self.p3s = []

        # novelty value of each arm in each play
        self.novelty1s = []
        self.novelty2s = []
        self.novelty3s = []

        # pragmatic value of each arm in each play
        self.pragmatic1s = []
        self.pragmatic2s = []
        self.pragmatic3s = []

        # estimated r in each play
        self.r1s = []
        self.r2s = []
        self.r3s = []

        # Selected arm index in each play
        self.selected_arms = []

        # Observed result in each play (win=1, loose=0)
        self.results = []

    def run(self):
        for i in range(self.n):
            self.run_one_step()

    def run_one_step(self):
        pragmatic1 = -self.r1 * np.log(self.C[0] / self.C[1]) - np.log(self.C[1])
        pragmatic2 = -self.r2 * np.log(self.C[0] / self.C[1]) - np.log(self.C[1])
        pragmatic3 = -self.r3 * np.log(self.C[0] / self.C[1]) - np.log(self.C[1])

        novelty1 = -1 / (2 * (self.a1win + self.a1loose))
        novelty2 = -1 / (2 * (self.a2win + self.a2loose))
        novelty3 = -1 / (2 * (self.a3win + self.a3loose))

        G = np.array([0.0, 0.0, 0.0])

        G[0] = novelty1 + pragmatic1
        G[1] = novelty2 + pragmatic2
        G[2] = novelty3 + pragmatic3

        p_pi = np.exp(-self.gamma * G) / np.sum(np.exp(-self.gamma * G))
        arm_index = np.random.choice(range(len(p_pi)), 1, p=p_pi)[0]  # np.argmax(G)
        result = np.random.binomial(1, self.true_prob[arm_index])

        self.results.append(result)
        self.selected_arms.append(arm_index)
        self.n_plays[arm_index] += 1
        self.n_wins[arm_index] += result
        self.n_looses[arm_index] += 1 - result

        # Update A matrix
        self.a1win = self.a0 + self.eta * self.n_wins[0]
        self.a1loose = self.a0 + self.eta * self.n_looses[0]
        self.a2win = self.a0 + self.eta * self.n_wins[1]
        self.a2loose = self.a0 + self.eta * self.n_looses[1]
        self.a3win = self.a0 + self.eta * self.n_wins[2]
        self.a3loose = self.a0 + self.eta * self.n_looses[2]

        self.r1 = self.a1win / (self.a1win + self.a1loose)
        self.r2 = self.a2win / (self.a2win + self.a2loose)
        self.r3 = self.a3win / (self.a3win + self.a3loose)

        # save the current values
        self.novelty1s.append(novelty1)
        self.novelty2s.append(novelty2)
        self.novelty3s.append(novelty3)

        self.pragmatic1s.append(pragmatic1)
        self.pragmatic2s.append(pragmatic2)
        self.pragmatic3s.append(pragmatic3)

        self.G1s.append(G[0])
        self.G2s.append(G[1])
        self.G3s.append(G[2])

        self.p1s.append(p_pi[0])
        self.p2s.append(p_pi[1])
        self.p3s.append(p_pi[2])

        self.r1s.append(self.r1)
        self.r2s.append(self.r2)
        self.r3s.append(self.r3)

        return arm_index, result

    def plot_G_each_arm(self):
        fig = plt.figure(figsize=(20.0, 4.0))
        ax1 = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1)
        ax2 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=1)
        ax3 = plt.subplot2grid((1, 3), (0, 2), rowspan=1, colspan=1)

        ax1.plot(range(self.n), self.novelty1s, label="novelty")
        ax1.plot(range(self.n), self.pragmatic1s, label="pragmatic")
        ax1.plot(range(self.n), self.G1s, label="G")
        ax1.plot()

        ax1.set_xlabel("Play count")
        # ax1.set_ylabel("G, novelty, pragmatic value in arm1")
        ax1.set_title("Arm1")

        ax2.plot(range(self.n), self.novelty2s, label="novelty")
        ax2.plot(range(self.n), self.pragmatic2s, label="pragmatic")
        ax2.plot(range(self.n), self.G2s, label="G")
        ax2.plot()

        ax2.set_xlabel("Play count")
        # ax2.set_ylabel("G, novelty, pragmatic value in arm2")
        ax2.set_title("Arm2")

        ax3.plot(range(self.n), self.novelty3s, label="novelty")
        ax3.plot(range(self.n), self.pragmatic3s, label="pragmatic")
        ax3.plot(range(self.n), self.G3s, label="G")
        ax3.plot()

        ax3.set_xlabel("Play count")
        # ax3.set_ylabel("G, novelty, pragmatic value in arm3")
        ax3.set_title("Arm3")

        handles, labels = ax3.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 0.9))

        st.pyplot(fig)

    def plot_p_pi(self):
        # G, Action probability p(pi), Estimated winning probability r in each arm

        fig = plt.figure(figsize=(20.0, 4.0))
        ax1 = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1)
        ax2 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=1)
        ax3 = plt.subplot2grid((1, 3), (0, 2), rowspan=1, colspan=1)

        ax1.plot(range(self.n), self.G1s, label="arm1")
        ax1.plot(range(self.n), self.G2s, label="arm2")
        ax1.plot(range(self.n), self.G3s, label="arm3")
        ax1.plot()

        ax1.set_xlabel("Play count")
        ax1.set_ylabel("G")
        ax1.set_title("G in each arm")

        ax2.plot(range(self.n), self.p1s, label="arm1")
        ax2.plot(range(self.n), self.p2s, label="arm2")
        ax2.plot(range(self.n), self.p3s, label="arm3")
        ax2.plot()

        ax2.set_xlabel("Play count")
        ax2.set_ylabel("p(pi)")
        ax2.set_ylim([0, 1])
        ax2.set_title("Action probability p(pi) in each arm")

        ax3.plot(range(self.n), self.r1s, label="arm1")
        ax3.plot(range(self.n), self.r2s, label="arm2")
        ax3.plot(range(self.n), self.r3s, label="arm3")
        ax3.plot()

        ax3.set_xlabel("Play count")
        ax3.set_ylabel("Estimateed r")
        ax3.set_ylim([0, 1])
        ax3.set_title("Estimated winning probability in each arm")

        handles, labels = ax3.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 0.9))

        st.pyplot(fig)

    def plot_reward(self):
        fig, ax = plt.subplots()
        plt.plot(range(self.n), [self.selected_arms[:i].count(0) for i in range(self.n)], label="arm1 select count")
        plt.plot(range(self.n), [self.selected_arms[:i].count(1) for i in range(self.n)], label="arm2 select count")
        plt.plot(range(self.n), [self.selected_arms[:i].count(2) for i in range(self.n)], label="arm3 select count")
        plt.plot(range(self.n), [sum(self.results[:i]) for i in range(self.n)], label="cumulative reward")
        plt.plot()
        plt.xlabel("Play count")
        plt.title("Cumulative reward and select count in each earm")
        plt.legend()
        st.pyplot(fig)


def main():
    st.header("Three-armed Bandit")
    st.markdown(
        """
        This interactive demo app of Three-armed Bandit to understand an agent behavior based on free energy principle.
        ## Three-armed bandit task
        - There are three bandit machines
        - Each bandit machine has own probability of winning
        - Agent doesn't know the true probability
        - Agent tries to maximize the reward for n trials

        ### Parameters
        - true_prob: True probability of winning in each bandit machine
        - rs: risk-seeking parameter in C matrix
        - 𝛾 : action precision parameter
        - a0 : concentration parameters of the beta distribution in a
        - 𝜂 : learning rate which controls the magnitude of updates in "A" matrix after each observation
        """
    )
    st.markdown("## Please set parameters")
    st.markdown("### True probability of winning in each bandit machine")
    p1 = st.number_input(
        "Probability of winning in bandit machine1", min_value=0.0, max_value=1.0, value=0.7, step=0.01, key="p1"
    )
    p2 = st.number_input(
        "Probability of winning in bandit machine2", min_value=0.0, max_value=1.0, value=0.6, step=0.01, key="p2"
    )
    p3 = st.number_input(
        "Probability of winning in bandit machine3", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="p3"
    )

    st.markdown("### Agent's parameters")
    rs = st.number_input(
        "rs : risk-seeking parameter in C matrix", min_value=0.0, max_value=100.0, value=1.0, step=0.01, key="rs"
    )

    gamma = st.number_input(
        "𝛾 : action precision parameter", min_value=0.0, max_value=100.0, value=1.0, step=0.01, key="gamma"
    )

    a0 = st.number_input(
        "a0 : concentration parameters of the beta distribution in a",
        min_value=0.01,
        max_value=100.0,
        value=1.0,
        step=0.01,
        key="a0",
    )

    eta = st.number_input(
        "𝜂 : learning rate which controls the magnitude of updates in 'A' matrix after each observation",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.01,
        key="eta",
    )

    n = st.number_input("n : number of play", min_value=1, max_value=100, value=30, step=1, key="n")

    button = st.button("Start Simulation")
    if button:
        fep = FEP(rs=rs, gamma=gamma, a0=a0, eta=eta, true_prob=np.array([p1, p2, p3]), n=n)
        fep.run()
        fep.plot_G_each_arm()
        fep.plot_p_pi()
        fep.plot_reward()


if __name__ == "__main__":
    main()
