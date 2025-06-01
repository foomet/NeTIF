import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib
import numpy as np

def num_packets_bar_plot():
    # Packet bins
    labels = list(range(1, 15))
    labels.append(r'$\geq15$')
    x = np.arange(len(labels))

    # Data (extracted from your PDF visual)
    cicids_values = [101807, 12175, 3505, 1589, 956, 984, 4191, 1114, 982, 1234, 1724, 1710, 1493, 2566, 20538]
    unsw_values   = [69326, 3047,  480,  237, 182, 201,  239,   55,  30,   13,   34,   45,   15,   13,   212]

    width = 0.35

    # Use the exact colors from your PDF:
    cicids_color = "#619CFF"     # Light blue
    unsw_color = "#00A86B"       # Reddish

    # Font configuration
    font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 30}
    matplotlib.rc('font', **font)

    # Create broken y-axis plot
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(18, 8), gridspec_kw={'height_ratios': [2, 3]})

    # Upper part (for large bars)
    ax1.bar(x - width/2, cicids_values, width, label='CICIDS2017', color=cicids_color)
    ax1.bar(x + width/2, unsw_values, width, label='UNSW-NB15', color=unsw_color)
    ax1.set_ylim(14000, 117000)
    ax1.set_yticks([14500, 50000, 100000])
    ax1.set_yticklabels(['14,500', '50,000', '100,000'])
    # ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
    # ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
    # ax1.ticklabel_format(style='scientific', axis='y', scilimits=(2, 2))

    # Lower part (for small bars)
    ax2.bar(x - width/2, cicids_values, width, color=cicids_color)
    ax2.bar(x + width/2, unsw_values, width, color=unsw_color)
    ax2.set_ylim(0, 5000)
    ax2.set_yticks([0, 1500, 3000, 4500])
    ax2.set_yticklabels(["0", "1,500","3,000", "4,500"])
    # ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
    # ax2.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    # ax2.ticklabel_format(style='scientific', axis='y')

    # X-axis labels
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Number of Packets per Flow", fontsize=30)
    # ax1.set_ylabel("Number of Flows", fontsize=22)
    ax2.set_ylabel("Number of Flows", fontsize=30)

    # Broken axis styling
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labeltop=False)
    ax2.tick_params(labeltop=False)

    d = .5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
                  linestyle='none', color='k', mec='k', mew=1, clip_on=False)
    ax1.plot(np.arange(-0.5, len(labels)), np.full(len(labels)+1, ax1.get_ylim()[0]), **kwargs)
    ax2.plot(np.arange(-0.5, len(labels)), np.full(len(labels)+1, ax2.get_ylim()[1]), **kwargs)


    # Add labels to top and bottom axis
    for i in range(len(x)):
        # Top axis: only label bars that are in the upper y-limit range
        if cicids_values[i] > 5000:
            ax1.text(x[i] - width/2, cicids_values[i] + 2000, f"{cicids_values[i]:,}", ha='center', va='bottom', fontsize=20)
        if unsw_values[i] > 5000:
            ax1.text(x[i] + width/2, unsw_values[i] + 2000, f"{unsw_values[i]:,}", ha='center', va='bottom', fontsize=20)

        # Bottom axis: label bars that are in the lower range
        if cicids_values[i] <= 5000:
            ax2.text(x[i] - width/2, cicids_values[i] + 100, f"{cicids_values[i]:,}", ha='center', va='bottom', fontsize=20)
        if unsw_values[i] <= 5000:
            ax2.text(x[i] + width/2, unsw_values[i] + 100, f"{unsw_values[i]:,}", ha='center', va='bottom', fontsize=20)

    plt.xlim(-0.65, 14.6)

    # Add legend and layout
    fig.legend(
        loc='upper right',
        bbox_to_anchor=(0.97, 0.96),  # adjust x (left–right), y (bottom–top)
        fontsize=30 ,
        framealpha=0.3
    )
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    matplotlib.rcParams['pdf.fonttype'] = 42  # Use TrueType fonts
    matplotlib.rcParams['ps.fonttype'] = 42
    # Save output
    plt.savefig("plots/recreated_figure3_broken_yaxis.pdf", bbox_inches='tight')
    plt.show()


def p_value_results_plots():
    # X-axis values: p values
    p_values = [1, 3, 5, 7]

    # Y-axis values: F1-scores for each dataset
    cicids_f1 = [1, 1, 1, 1]  # Example values
    unsw_f1 = [0.86, 0.71, 0.76, 0.62]  # Example values

    # Font configuration
    font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 25}
    matplotlib.rc('font', **font)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(p_values, unsw_f1, marker='o', label='UNSW')
    plt.plot(p_values, cicids_f1, marker='s', label='CICIDS')
    plt.xticks([1, 2, 3, 4, 5, 6, 7])
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.xlabel('Parameter p')
    plt.ylabel('F1-score')
    plt.grid(True)
    plt.legend(loc='lower left',
                fontsize=23,
                framealpha=0.3,
                bbox_to_anchor=(0.01, 0.01),  # adjust x (left–right), y (bottom–top)
                )
    plt.tight_layout()
    matplotlib.rcParams['pdf.fonttype'] = 42  # Use TrueType fonts
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.savefig("plots/f1_scores_varying_p.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def Figure_5():
    # Classes
    classes = ['Benign', 'DDoS', 'DoS Hulk', 'FTP-Patator', 'SSH-Patator']
    x = np.arange(len(classes))

    # Width of each bar
    bar_width = 0.2

    # F1-scores
    f1_1 = [0.29, 0.00, 0.10, 0.32, 0.34]  # Their transform + their arch
    f1_2 = [0.41, 0.00, 0.00, 0.00, 0.00]  # Their transform + our arch
    f1_3 = [1.00, 1.00, 1.00, 1.00, 1.00]  # Our transform + their arch
    f1_4 = [1.00, 1.00, 1.00, 1.00, 1.00]  # Our transform + our arch

    # Labels and colors
    labels = [
        "CNN-Trans Transform + CNN-Trans Architecture",
        "CNN-Trans Transform + NeTIF Architecture",
        "NeTIF Transform + CNN-Trans Architecture",
        "NeTIF Transform + NeTIF Architecture"
    ]
    colors = ['lightcoral', 'lightskyblue', 'orange', 'mediumseagreen']

    # Create figure
    plt.figure(figsize=(10, 5))

    # Font configuration
    font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 20}
    matplotlib.rc('font', **font)

    # Plot each configuration
    plt.bar(x - 1.5 * bar_width, f1_1, width=bar_width, label=labels[0], color=colors[0])
    plt.bar(x - 0.5 * bar_width, f1_2, width=bar_width, label=labels[1], color=colors[1])
    plt.bar(x + 0.5 * bar_width, f1_3, width=bar_width, label=labels[2], color=colors[2])
    plt.bar(x + 1.5 * bar_width, f1_4, width=bar_width, label=labels[3], color=colors[3])

    # Axis settings
    plt.xticks(x, classes, rotation=0, fontsize=14)
    plt.ylabel("F1-score", fontsize=20)
    plt.ylim(0, 1.1)

    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False,
        fontsize=15
    )
    plt.tight_layout()

    matplotlib.rcParams['pdf.fonttype'] = 42  # Use TrueType fonts
    matplotlib.rcParams['ps.fonttype'] = 42

    plt.savefig("plots/transform_comparison_bar.pdf", format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # num_packets_bar_plot()
    # p_value_results_plots()
    Figure_5()