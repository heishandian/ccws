import matplotlib.pyplot as plt
from cycler import cycler


def plot_lr_curve(bath_path, rewards, tasks, episode_rur, episode_rbd, alg):
    num_colors = 4
    cm = plt.get_cmap('gist_rainbow')

    fig = plt.figure(figsize=(12, 10))

    ax = fig.add_subplot(221)
    colors = [cm(1. * i / num_colors) for i in range(num_colors)]
    ax.set_prop_cycle(cycler('color', colors))
    ax.plot(rewards, linewidth=2, label=alg)
    plt.legend(loc=1)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Cumulative Rewards", fontsize=20)

    ax = fig.add_subplot(222)
    colors = [cm(1. * i / num_colors) for i in range(num_colors)]
    ax.set_prop_cycle(cycler('color', colors))
    ax.plot(tasks, linewidth=2, label=alg)
    plt.legend(loc=1)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Number of tasks", fontsize=20)

    ax = fig.add_subplot(223)
    colors = [cm(1. * i / num_colors) for i in range(num_colors)]
    ax.set_prop_cycle(cycler('color', colors))
    ax.plot(episode_rur, linewidth=2, label=alg)
    plt.legend(loc=1)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("RUR", fontsize=20)

    ax = fig.add_subplot(224)
    colors = [cm(1. * i / num_colors) for i in range(num_colors)]
    ax.set_prop_cycle(cycler('color', colors))
    ax.plot(episode_rbd, linewidth=2, label=alg)
    plt.legend(loc=1)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("RID", fontsize=20)

    plt.savefig(bath_path + "_lr_curve" + ".pdf")
    plt.cla()
