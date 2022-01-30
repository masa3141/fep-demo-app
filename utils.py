import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import streamlit as st


# def create_graph():
#     # 描画領域を用意する
#     fig = plt.figure()
#     ax = fig.add_subplot()
#     # ランダムな値をヒストグラムとしてプロットする
#     x = np.random.normal(loc=0.0, scale=1.0, size=(100,))
#     ax.hist(x, bins=20)
#     # Matplotlib の Figure を指定して可視化する
#     st.pyplot(fig)


# if st.button("Top button"):
#     create_graph()


def plot_likelihood(matrix, xlabels=list(range(9)), ylabels=list(range(9)), title_str="Likelihood distribution (A)"):
    """
    Plots a 2-D likelihood matrix as a heatmap
    """

    if not np.isclose(matrix.sum(axis=0), 1.0).all():
        raise ValueError(
            "Distribution not column-normalized! Please normalize (ensure matrix.sum(axis=0) == 1.0 for all columns)"
        )

    fig = plt.figure(figsize=(6, 6))
    ax = sns.heatmap(matrix, xticklabels=xlabels, yticklabels=ylabels, cmap="gray", cbar=False, vmin=0.0, vmax=1.0)
    plt.title(title_str)
    plt.show()


def plot_grid(grid_locations, num_x=3, num_y=3):
    """
    Plots the spatial coordinates of GridWorld as a heatmap, with each (X, Y) coordinate
    labeled with its linear index (its `state id`)
    """
    # Rendering Matplotlib AxesSubplots in Streamlit
    # https://discuss.streamlit.io/t/rendering-matplotlib-axessubplots-in-streamlit/5662/4
    fig, ax = plt.subplots()
    grid_heatmap = np.zeros((num_x, num_y))
    for linear_idx, location in enumerate(grid_locations):
        y, x = location
        grid_heatmap[y, x] = linear_idx
    sns.set(font_scale=1.5)
    ax = sns.heatmap(grid_heatmap, annot=True, cbar=False, fmt=".0f", cmap="crest")
    # Matplotlib の Figure を指定して可視化する
    st.pyplot(fig)


def plot_point_on_grid(state_vector, grid_locations):
    """
    Plots the current location of the agent on the grid world
    """
    state_index = np.where(state_vector)[0][0]
    y, x = grid_locations[state_index]
    grid_heatmap = np.zeros((3, 3))
    grid_heatmap[y, x] = 1.0
    sns.heatmap(grid_heatmap, cbar=False, fmt=".0f")


def plot_beliefs(belief_dist, title_str=""):
    """
    Plot a categorical distribution or belief distribution, stored in the 1-D numpy vector `belief_dist`
    """

    if not np.isclose(belief_dist.sum(), 1.0):
        raise ValueError("Distribution not normalized! Please normalize")

    plt.grid(zorder=0)
    plt.bar(range(belief_dist.shape[0]), belief_dist, color="r", zorder=3)
    plt.xticks(range(belief_dist.shape[0]))
    plt.title(title_str)
    plt.show()


# for i in range(5):
#     create_graph()
#     time.sleep(1)


# status_area = st.empty()

# # カウントダウン
# count_down_sec = 5
# for i in range(count_down_sec):
#     # プレースホルダーに残り秒数を書き込む
#     status_area.write(f"{count_down_sec - i} sec left")
#     # スリープ処理を入れる
#     time.sleep(1)

# # 完了したときの表示
# status_area.write("Done!")
# # 風船飛ばす
# st.balloons()

# status_text = st.empty()
# # プログレスバー
# progress_bar = st.progress(0)

# for i in range(100):
#     status_text.text(f"Progress: {i}%")
#     # for ループ内でプログレスバーの状態を更新する
#     progress_bar.progress(i + 1)
#     time.sleep(0.1)

# status_text.text("Done!")
# st.balloons()
