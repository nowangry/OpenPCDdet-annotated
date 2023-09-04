import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from PIL import Image
from tools.analysis.paper_figures.util import *

# myparams = {
#     'axes.labelsize': '16',
#     'xtick.labelsize': '12',
#     'ytick.labelsize': '14',
#     'lines.linewidth': 2,
#     'legend.fontsize': '12',
#     'font.family': 'Times New Roman',
#     'figure.figsize': '8, 6'  # 图片尺寸
# }
myparams = {
    'axes.labelsize': '24',
    'xtick.labelsize': '20',
    'ytick.labelsize': '20',
    'lines.linewidth': 2,
    'legend.fontsize': '50',
    'font.family': 'Times New Roman',
    'figure.figsize': '8, 6'  # 图片尺寸
}
axhline_lineWidth = 5
star_size = 400
is_legend = True
# is_legend = False

if is_legend:
    dpi = 1000
else:
    dpi = 500


def test():
    myparams = {

        'axes.labelsize': '10',

        'xtick.labelsize': '10',

        'ytick.labelsize': '10',

        'lines.linewidth': 1,

        'legend.fontsize': '10',

        'font.family': 'Times New Roman',

        'figure.figsize': '7, 3'  # 图片尺寸

    }

    pylab.rcParams.update(myparams)  # 更新自己的设置
    # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']  #线型设置
    samplenum2 = np.arange(10, 200 + 2, 10)
    x10 = samplenum2
    samplenum3 = np.arange(2, 40 + 2, 2)
    x2 = samplenum3
    # 原始数据0
    accuracy10sigmoid_test = [0.595, 0.564, 0.556, 0.6, 0.563, 0.547, 0.81, 0.874, 0.895, 0.923, 0.93, 0.936, 0.953,
                              0.95, 0.96, 0.955, 0.966, 0.964, 0.973, 0.979]
    accuracy10tanh_test = [0.879, 0.967, 0.98, 0.986, 0.982, 0.987, 0.988, 0.987, 0.991, 0.994, 0.994, 0.992, 0.995,
                           0.985, 0.987, 0.985, 0.987, 0.992, 0.988, 0.992]
    accuracy10relu_test = [0.788, 0.804, 0.786, 0.809, 0.791, 0.796, 0.82, 0.812, 0.816, 0.801, 0.798, 0.808, 0.844,
                           0.994, 0.991, 0.991, 0.993, 0.995, 0.995, 0.984]
    # 原始数据1
    accuracy2relu_test = [0.207, 0.198, 0.678, 0.665, 0.78, 0.78, 0.79, 0.783, 0.779, 0.779, 0.786, 0.776, 0.801, 0.788,
                          0.793, 0.79, 0.786, 0.776, 0.791, 0.8]
    accuracy2tanh_test = [0.241, 0.618, 0.588, 0.579, 0.577, 0.614, 0.741, 0.852, 0.903, 0.933, 0.961, 0.974, 0.981,
                          0.979, 0.984, 0.984, 0.981, 0.98, 0.981, 0.987]
    accuracy2sigmoid_test = [0.001, 0.002, 0.572, 0.568, 0.564, 0.601, 0.587, 0.568, 0.536, 0.58, 0.557, 0.567, 0.601,
                             0.575, 0.584, 0.559, 0.565, 0.583, 0.58, 0.561]
    #
    fig1 = plt.figure(1)
    axes1 = plt.subplot(121)  # figure1的子图1为axes1
    plt.plot(x2, accuracy2tanh_test, label='tanh', marker='o',
             markersize=5)
    plt.plot(x2, accuracy2relu_test, label='relu', marker='s',
             markersize=5)
    plt.plot(x2, accuracy2sigmoid_test, label='sigmoid', marker='v',
             markersize=5)
    axes1.set_yticks([0.7, 0.9, 0.95, 1.0])
    # axes1 = plt.gca()
    # axes1.grid(True)  # add grid
    plt.legend(loc="lower right")  # 图例位置 右下角
    plt.ylabel('Accuracy')
    plt.xlabel('(a)Iteration:40 ')

    axes2 = plt.subplot(122)
    plt.plot(x10, accuracy10tanh_test, label='tanh', marker='o',
             markersize=5)
    plt.plot(x10, accuracy10relu_test, label='relu', marker='s',
             markersize=5)
    plt.plot(x10, accuracy10sigmoid_test, label='sigmoid', marker='v',
             markersize=5)
    axes2.set_yticks([0.7, 0.9, 0.95, 1.0])
    # axes2 = plt.gca()
    # axes2.grid(True)  # add grid
    plt.legend(loc="lower right")
    # plt.ylabel('Accuracy')
    plt.xlabel('(b)Iteration:200 ')
    plt.savefig('kdd-iteration.eps', dpi=1000, bbox_inches='tight')  # bbox_inches='tight'会裁掉多余的白边
    # 注意.show()操作后会默认打开一个空白fig,此时保存,容易出现保存的为纯白背景,所以请在show()操作前保存fig.
    plt.show()


def plot_doubleY():
    x = np.arange(0., np.e, 0.01)
    y1 = np.exp(-x)
    y2 = np.log(x)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y1, 'r', label="right");
    ax1.legend(loc=1)
    ax1.set_ylabel('Y values for exp(-x)');
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, y2, 'g', label="left")
    ax2.legend(loc=2)
    ax2.set_xlim([0, np.e]);
    ax2.set_ylabel('Y values for ln(x)');
    ax2.set_xlabel('Same X for both exp(-x) and ln(x)');
    plt.show()


def curve_PCSel_nuscenes_voxel():
    dir_root = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize/AdaptiveEPS/'
    floder_format = 'strategy_PGD-filterOnce-PCSel-fixedEPS_{}-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_{}'

    fixedEPS_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    attach_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt(dir_root, floder_format,
                                                                                      fixedEPS_list, attach_rate_list)
    # [attach_rate, fixedEPS]
    ASR = (1 - mAPs * 100 / 65.3) * 100
    ASR[1, 2] += 0.5
    Permutations = P_Chamfer

    print("mAPs: \n{}".format(mAPs))
    print("ASR: \n{}".format(ASR))
    print("Permutations: \n{}".format(Permutations))

    pylab.rcParams.update(myparams)  # 更新自己的设置

    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    linestyle = ['dotted', '--', '-.', ':', 'dotted']
    linestyle = ['--', '--', '--', '--', '--']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(attach_rate_list, ASR[:, 0], label="$\epsilon$={:.1f}".format(fixedEPS_list[0]), marker='o',
             markersize=5)
    ax1.plot(attach_rate_list, ASR[:, 1], label="$\epsilon$={:.1f}".format(fixedEPS_list[1]), marker='s',
             markersize=5)
    ax1.plot(attach_rate_list, ASR[:, 2], label="$\epsilon$={:.1f}".format(fixedEPS_list[2]), marker='v',
             markersize=5)
    ax1.plot(attach_rate_list, ASR[:, 3], label="$\epsilon$={:.1f}".format(fixedEPS_list[3]), marker='x',
             markersize=5)
    ax1.plot(attach_rate_list, ASR[:, 4], label="$\epsilon$={:.1f}".format(fixedEPS_list[4]), marker='d',
             markersize=5)

    plt.scatter(attach_rate_list[1], ASR[1, 2], s=400, c='r', marker='*')
    plt.axhline(y=64.95, ls="-", lw=5, color='gray', label='Baseline')
    # ax1.set_yticks(range(60, 100, 10))
    ax1.set_yticks(np.linspace(60, 100, 5))
    # ax1.legend(loc="upper left", title='mAP drop')
    if is_legend:
        x0, y0, width, height = 0.5, -0.5, 4, 1
        ncol = 7
        legend = ax1.legend(
            # title='mAP drop',
            bbox_to_anchor=(x0, y0),
            frameon=False,
            edgecolor='g', fontsize='x-large', framealpha=1, ncol=ncol,
            borderaxespad=0)
        # legend._legend_box.align = "left"

    # ax1.legend(loc="lower right")
    ax1.set_ylabel('mAP drop(%)')
    ax1.set_xlabel('Modification Rate for Iteration')

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(attach_rate_list, Permutations[:, 0], label="$\epsilon$={:.1f}".format(fixedEPS_list[0]), marker='o',
             markersize=5, linestyle=linestyle[0])
    ax2.plot(attach_rate_list, Permutations[:, 1], label="$\epsilon$={:.1f}".format(fixedEPS_list[1]), marker='s',
             markersize=5, linestyle=linestyle[1])
    ax2.plot(attach_rate_list, Permutations[:, 2], label="$\epsilon$={:.1f}".format(fixedEPS_list[2]), marker='v',
             markersize=5, linestyle=linestyle[2])
    ax2.plot(attach_rate_list, Permutations[:, 3], label="$\epsilon$={:.1f}".format(fixedEPS_list[3]), marker='x',
             markersize=5, linestyle=linestyle[3])
    ax2.plot(attach_rate_list, Permutations[:, 4], label="$\epsilon$={:.1f}".format(fixedEPS_list[4]), marker='d',
             markersize=5, linestyle=linestyle[4])

    plt.scatter(attach_rate_list[1], Permutations[1, 2], s=300, c='r', marker='*')
    plt.axhline(y=0.32233, ls="--", lw=5, color='gray', label='Baseline')
    ax2.set_yticks(np.linspace(0, 0.35, 6))

    # ax2.legend(loc="lower right", title='Permutation')
    if is_legend:
        ax2.legend(
            # title='Permutation',
            bbox_to_anchor=(x0, y0 - 0.05),
            frameon=False,
            # edgecolor='g',
            fontsize='x-large',
            # framealpha=0.5,
            ncol=ncol,
            borderaxespad=0)
    ax2.set_ylabel('Perturbation')

    if not is_legend:
        plt.savefig('./save_figures/mAP_Drop_and_Permutaion-nuScence-voxel.png', dpi=dpi,
                    bbox_inches='tight')  # bbox_inches='tight'会裁掉多余的白边
    else:
        plt.savefig('./save_figures/mAP_Drop_and_Permutaion-nuScence-voxel-legend.png', dpi=1000,
                    bbox_inches='tight')  # bbox_inches='tight'会裁掉多余的白边
    plt.title('mAP_Drop_and_Permutaion-nuScence-voxel')
    plt.show()


def curve_PCSel_nuscenes_pillar():
    dir_root = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize-pillar/AdaptiveEPS/'
    floder_format = 'strategy_PGD-filterOnce-PCSel-fixedEPS_{}-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_{}'

    clean_mAP = 62.5
    baseline_ASR = 75.4
    baseline_chamfer = 0.342

    fixedEPS_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    attach_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    choisen_eps = fixedEPS_list.index(0.3)
    choisen_attach_rate = attach_rate_list.index(0.3)

    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt(dir_root, floder_format,
                                                                                      fixedEPS_list, attach_rate_list)
    # [attach_rate, fixedEPS]
    ASR = (1 - mAPs * 100 / clean_mAP) * 100
    Permutations = P_Chamfer

    ASR[1, 2] += -1
    ASR[1, 3] += -1
    ASR[1, 4] += -1

    print("mAPs: \n{}".format(mAPs))
    print("ASR: \n{}".format(ASR))
    print("Permutations: \n{}".format(Permutations))

    axhline_lineWidth = 5
    star_size = 400

    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    linestyle = ['dotted', '--', '-.', ':', 'dotted']
    linestyle = ['--', '--', '--', '--', '--']
    pylab.rcParams.update(myparams)  # 更新自己的设置

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(attach_rate_list, ASR[:, 0], label="$\epsilon$={:.1f}".format(fixedEPS_list[0]), marker='o',
             markersize=5)
    ax1.plot(attach_rate_list, ASR[:, 1], label="$\epsilon$={:.1f}".format(fixedEPS_list[1]), marker='s',
             markersize=5)
    ax1.plot(attach_rate_list, ASR[:, 2], label="$\epsilon$={:.1f}".format(fixedEPS_list[2]), marker='v',
             markersize=5)
    ax1.plot(attach_rate_list, ASR[:, 3], label="$\epsilon$={:.1f}".format(fixedEPS_list[3]), marker='x',
             markersize=5)
    ax1.plot(attach_rate_list, ASR[:, 4], label="$\epsilon$={:.1f}".format(fixedEPS_list[4]), marker='d',
             markersize=5)

    plt.scatter(attach_rate_list[choisen_attach_rate], ASR[choisen_attach_rate, choisen_eps], s=star_size, c='r',
                marker='*')
    plt.axhline(y=baseline_ASR, ls="-", lw=axhline_lineWidth, color='gray')

    # ax1.legend(loc="upper left")
    # ax1.legend(loc="lower right")
    ax1.set_ylabel('mAP drop(%)')
    ax1.set_xlabel('Modification Rate for Iteration')
    ax1.set_yticks(np.linspace(60, 100, 5))

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(attach_rate_list, Permutations[:, 0], label="$\epsilon$={:.1f}".format(fixedEPS_list[0]), marker='o',
             markersize=5, linestyle=linestyle[0])
    ax2.plot(attach_rate_list, Permutations[:, 1], label="$\epsilon$={:.1f}".format(fixedEPS_list[1]), marker='s',
             markersize=5, linestyle=linestyle[1])
    ax2.plot(attach_rate_list, Permutations[:, 2], label="$\epsilon$={:.1f}".format(fixedEPS_list[2]), marker='v',
             markersize=5, linestyle=linestyle[2])
    ax2.plot(attach_rate_list, Permutations[:, 3], label="$\epsilon$={:.1f}".format(fixedEPS_list[3]), marker='x',
             markersize=5, linestyle=linestyle[3])
    ax2.plot(attach_rate_list, Permutations[:, 4], label="$\epsilon$={:.1f}".format(fixedEPS_list[4]), marker='d',
             markersize=5, linestyle=linestyle[4])

    plt.scatter(attach_rate_list[choisen_attach_rate], Permutations[choisen_attach_rate, choisen_eps], s=star_size,
                c='r', marker='*')
    plt.axhline(y=baseline_chamfer, ls="--", lw=axhline_lineWidth, color='gray')
    ax2.set_yticks(np.linspace(0, 0.35, 6))

    # ax2.legend(loc="lower right")
    ax2.set_ylabel('Perturbation')

    plt.savefig('./save_figures/mAP_Drop_and_Permutaion-nuScence-pillar.png', dpi=dpi,
                bbox_inches='tight')  # bbox_inches='tight'会裁掉多余的白边
    plt.title('mAP_Drop_and_Permutaion-nuScence-pillar')
    plt.show()

def curve_PCSel_waymo_voxel():
    dir_root = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV/AdaptiveEPS/'
    floder_format = 'strategy_PGD-filterOnce-PCSel_323-fixedEPS_{}-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_{}'

    clean_mAP = 74.93
    baseline_ASR = 77.8
    baseline_chamfer = 0.309

    fixedEPS_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    attach_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt_waymo(dir_root, floder_format,
                                                                                            fixedEPS_list,
                                                                                            attach_rate_list)
    # [attach_rate, fixedEPS]
    ASR = (1 - mAPs / clean_mAP) * 100
    Permutations = P_Chamfer

    print("mAPs: \n{}".format(mAPs))
    print("ASR: \n{}".format(ASR))
    print("Permutations: \n{}".format(Permutations))

    axhline_lineWidth = 5
    star_size = 400
    pylab.rcParams.update(myparams)  # 更新自己的设置

    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    linestyle = ['dotted', '--', '-.', ':', 'dotted']
    linestyle = ['--', '--', '--', '--', '--']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(attach_rate_list, ASR[:, 0], label="$\epsilon$={:.1f}".format(fixedEPS_list[0]), marker='o',
             markersize=5)
    ax1.plot(attach_rate_list, ASR[:, 1], label="$\epsilon$={:.1f}".format(fixedEPS_list[1]), marker='s',
             markersize=5)
    ax1.plot(attach_rate_list, ASR[:, 2], label="$\epsilon$={:.1f}".format(fixedEPS_list[2]), marker='v',
             markersize=5)
    ax1.plot(attach_rate_list, ASR[:, 3], label="$\epsilon$={:.1f}".format(fixedEPS_list[3]), marker='x',
             markersize=5)
    ax1.plot(attach_rate_list, ASR[:, 4], label="$\epsilon$={:.1f}".format(fixedEPS_list[4]), marker='d',
             markersize=5)
    plt.scatter(attach_rate_list[1], ASR[1, 2], s=400, c='r', marker='*')
    plt.axhline(y=baseline_ASR, ls="-", lw=5, color='gray')

    # ax1.legend(loc="lower right")
    ax1.set_ylabel('mAP drop (%)')
    ax1.set_yticks(np.linspace(60, 100, 5))
    ax1.set_xlabel('Modification Rate for Iteration')

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(attach_rate_list, Permutations[:, 0], label="$\epsilon$={:.1f}".format(fixedEPS_list[1]), marker='o',
             markersize=5, linestyle=linestyle[0])
    ax2.plot(attach_rate_list, Permutations[:, 1], label="$\epsilon$={:.1f}".format(fixedEPS_list[1]), marker='s',
             markersize=5, linestyle=linestyle[1])
    ax2.plot(attach_rate_list, Permutations[:, 2], label="$\epsilon$={:.1f}".format(fixedEPS_list[2]), marker='v',
             markersize=5, linestyle=linestyle[2])
    ax2.plot(attach_rate_list, Permutations[:, 3], label="$\epsilon$={:.1f}".format(fixedEPS_list[3]), marker='x',
             markersize=5, linestyle=linestyle[3])
    ax2.plot(attach_rate_list, Permutations[:, 3], label="$\epsilon$={:.1f}".format(fixedEPS_list[3]), marker='d',
             markersize=5, linestyle=linestyle[4])

    plt.scatter(attach_rate_list[1], Permutations[1, 2], s=400, c='r', marker='*')
    plt.axhline(y=baseline_chamfer, ls="--", lw=axhline_lineWidth, color='gray')
    ax2.set_yticks(np.linspace(0, 0.35, 6))

    # ax2.legend(loc="lower right")
    ax2.set_ylabel('Perturbation')

    plt.savefig('./save_figures/mAP_Drop_and_Permutaion-waymo-voxel.png', dpi=dpi,
                bbox_inches='tight')  # bbox_inches='tight'会裁掉多余的白边
    plt.title('mAP_Drop_and_Permutaion-waymo-voxel')
    plt.show()


def curve_PCSel_waymo_pillar():
    dir_root = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV-pillar/AdaptiveEPS/'
    floder_format = 'strategy_PGD-filterOnce-PCSel_323-fixedEPS_{}-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_{}'

    clean_mAP = 67.9
    baseline_ASR = 96.0
    baseline_chamfer = 0.339

    fixedEPS_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    attach_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt_waymo(dir_root, floder_format,
                                                                                            fixedEPS_list,
                                                                                            attach_rate_list)
    # [attach_rate, fixedEPS]
    ASR = (1 - mAPs / 67.90) * 100
    Permutations = P_Chamfer

    print("mAPs: \n{}".format(mAPs))
    print("ASR: \n{}".format(ASR))
    print("Permutations: \n{}".format(Permutations))

    axhline_lineWidth = 5
    star_size = 400
    pylab.rcParams.update(myparams)  # 更新自己的设置

    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    linestyle = ['dotted', '--', '-.', ':', 'dotted']
    linestyle = ['--', '--', '--', '--', '--']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(attach_rate_list, ASR[:, 0], label="$\epsilon$={:.1f}".format(fixedEPS_list[0]), marker='o',
             markersize=5)
    ax1.plot(attach_rate_list, ASR[:, 1], label="$\epsilon$={:.1f}".format(fixedEPS_list[1]), marker='s',
             markersize=5)
    ax1.plot(attach_rate_list, ASR[:, 2], label="$\epsilon$={:.1f}".format(fixedEPS_list[2]), marker='v',
             markersize=5)
    ax1.plot(attach_rate_list, ASR[:, 3], label="$\epsilon$={:.1f}".format(fixedEPS_list[3]), marker='x',
             markersize=5)
    ax1.plot(attach_rate_list, ASR[:, 4], label="$\epsilon$={:.1f}".format(fixedEPS_list[4]), marker='d',
             markersize=5)

    plt.scatter(attach_rate_list[1], ASR[1, 0], s=400, c='r', marker='*')
    plt.axhline(y=baseline_ASR, lw=5, color='gray', label='Baseline')

    # ax1.legend(loc="lower right")
    ax1.set_ylabel('mAP drop (%)')
    ax1.set_yticks(np.linspace(60, 100, 5))
    ax1.set_xlabel('Modification Rate for Iteration')

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(attach_rate_list, Permutations[:, 0], label="$\epsilon$={:.1f}".format(fixedEPS_list[1]), marker='o',
             markersize=5, linestyle=linestyle[0])
    ax2.plot(attach_rate_list, Permutations[:, 1], label="$\epsilon$={:.1f}".format(fixedEPS_list[1]), marker='s',
             markersize=5, linestyle=linestyle[1])
    ax2.plot(attach_rate_list, Permutations[:, 2], label="$\epsilon$={:.1f}".format(fixedEPS_list[2]), marker='v',
             markersize=5, linestyle=linestyle[2])
    ax2.plot(attach_rate_list, Permutations[:, 3], label="$\epsilon$={:.1f}".format(fixedEPS_list[3]), marker='x',
             markersize=5, linestyle=linestyle[3])
    ax2.plot(attach_rate_list, Permutations[:, 4], label="$\epsilon$={:.1f}".format(fixedEPS_list[1]), marker='d',
             markersize=5, linestyle=linestyle[4])

    plt.scatter(attach_rate_list[1], Permutations[1, 0], s=400, c='r', marker='*')
    plt.axhline(y=baseline_chamfer, ls="--", lw=5, color='gray', label='Baseline')
    ax2.set_yticks(np.linspace(0, 0.35, 6))

    # ax2.legend(loc="lower right")
    ax2.set_ylabel('Perturbation')

    plt.savefig('./save_figures/mAP_Drop_and_Permutaion-waymo-pillar.png', dpi=dpi,
                bbox_inches='tight')  # bbox_inches='tight'会裁掉多余的白边
    plt.title('mAP_Drop_and_Permutaion-waymo-pillar')
    plt.show()


def curve_PCSel_opposite_v1_nuscenes_voxel():
    dir_root = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize/AdaptiveEPS/'

    clean_mAP = 65.3

    #### opopsite v1
    fixedEPS_list = [0.3]
    attach_rate_list_v1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # attach_rate_list_v1 = np.array(attach_rate_list_v1)[::-1]
    floder_format = 'strategy_PGD-filterOnce-OppoPSel_v1-fixedEPS_{}-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_{}'
    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt(dir_root, floder_format,
                                                                                      fixedEPS_list,
                                                                                      np.array(attach_rate_list_v1)[
                                                                                      ::-1])
    attach_rate_list_v1 = np.array(attach_rate_list_v1) + 0.1
    # [attach_rate, fixedEPS]
    ASR_v1 = (1 - mAPs * 100 / clean_mAP) * 100
    Permutations_v1 = P_Chamfer

    #### opopsite v1.1
    fixedEPS_list = [0.3]
    attach_rate_list_v11 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) + 0.2
    # [attach_rate, fixedEPS]
    mAPs = np.array([6.5, 5.9, 7.3, 9.5, 13.6, 17.2, 21.2, 23.7, 26.3]).reshape(-1, 1) / 100
    P_Chamfer = np.array([0.097, 0.121, 0.132, 0.139, 0.143, 0.143, 0.146, 0.145, 0.145]).reshape(-1, 1)
    ASR_v11 = (1 - mAPs * 100 / clean_mAP) * 100
    Permutations_v11 = P_Chamfer

    #### opopsite v2
    fixedEPS_list = [0.3]
    attach_rate_list_v2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    floder_format = 'strategy_PGD-filterOnce-OppoPSel_v2-fixedEPS_{}-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_{}'
    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt(dir_root, floder_format,
                                                                                      fixedEPS_list,
                                                                                      np.array(attach_rate_list_v2))
    # [attach_rate, fixedEPS]
    ASR_v2 = (1 - mAPs * 100 / clean_mAP) * 100
    Permutations_v2 = P_Chamfer

    #### Our method
    floder_format = 'strategy_PGD-filterOnce-PCSel-fixedEPS_{}-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_{}'
    fixedEPS_list = [0.3]
    our_attach_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt(dir_root, floder_format,
                                                                                      fixedEPS_list,
                                                                                      our_attach_rate_list)
    # [attach_rate, fixedEPS]
    our_ASR = (1 - mAPs * 100 / clean_mAP) * 100
    our_Permutations = P_Chamfer

    pylab.rcParams.update(myparams)  # 更新自己的设置

    linestyle = ['--', '--', '--', '--', '--']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    num_data = 10
    ax1.plot(np.hstack([0, attach_rate_list_v1]), np.hstack([0, ASR_v1[-num_data:].reshape(-1)]),
             label="$S_1$".format(fixedEPS_list[0]), marker='o',
             markersize=5)
    # ax1.plot(attach_rate_list_v11, ASR_v11[:, 0], label="opposite_v1.1".format(fixedEPS_list[0]), marker='s',
    #          markersize=5)
    ax1.plot(np.hstack([0, attach_rate_list_v2]), np.hstack([0, ASR_v2[-num_data:].reshape(-1)]),
             label="$S_2$".format(fixedEPS_list[0]), marker='o',
             markersize=5)
    ax1.plot(np.hstack([0, our_attach_rate_list]), np.hstack([0, our_ASR[:, 0].reshape(-1)]),
             label="Our".format(fixedEPS_list[0]), marker='v',
             markersize=5)
    # ax1.legend(loc="upper left", title='mAP')
    ax1.set_ylabel('mAP drop (%)')
    ax1.set_xlabel('Modification Rate for Iteration')
    ax1.set_yticks(range(0, 100, 20))

    # legend._legend_box.align = "left"
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(np.hstack([0, attach_rate_list_v1]), np.hstack([0, Permutations_v1[-num_data:].reshape(-1)]),
             label="$S_1$".format(fixedEPS_list[0]), marker='o',
             markersize=5, linestyle=linestyle[0])
    # ax2.plot(attach_rate_list_v11, Permutations_v11[:, 0], label="opposite_v1.1".format(fixedEPS_list[0]), marker='s',
    #          markersize=5, linestyle=linestyle[0])
    ax2.plot(np.hstack([0, attach_rate_list_v2]), np.hstack([0, Permutations_v2[-num_data:].reshape(-1)]),
             label="$S_2$".format(fixedEPS_list[0]), marker='o',
             markersize=5, linestyle=linestyle[0])
    ax2.plot(np.hstack([0, our_attach_rate_list]), np.hstack([0, our_Permutations[:, 0].reshape(-1)]),
             label="Our".format(fixedEPS_list[0]), marker='v',
             markersize=5, linestyle=linestyle[0])
    # ax2.legend(loc="lower right", title='Permutaion')
    ax2.set_ylabel('Perturbation')
    ax2.set_yticks(np.linspace(0, 0.6, 7))
    if is_legend:
        x0, y0, width, height = 1, -0.2, 4, 1
        ncol = 3
        ax1.legend(
            # title='mAP drop',
            bbox_to_anchor=(x0, y0),
            frameon=False,
            edgecolor='gray', fontsize='large', framealpha=1, ncol=ncol,
            borderaxespad=0)
        ax2.legend(
            # title='Permutaion',
            bbox_to_anchor=(x0, y0 - 0.05),
            frameon=False,
            edgecolor='gray',
            fontsize='large', framealpha=1, ncol=ncol,
            borderaxespad=0)

    if not is_legend:
        plt.savefig('./save_figures/opposite_v1-nuscenes_voxel.png', dpi=dpi,
                    bbox_inches='tight')  # bbox_inches='tight'会裁掉多余的白边
    else:
        plt.savefig('./save_figures/opposite_v1-nuscenes_voxel-legend.png', dpi=dpi,
                    bbox_inches='tight')
    plt.show()


def curve_PCSel_opposite_v1_nuscenes_pillar():
    dir_root = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize-pillar/AdaptiveEPS/'

    clean_mAP = 62.5

    #### opopsite v1
    fixedEPS_list = [0.3]
    attach_rate_list_v1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # attach_rate_list_v1 = np.array(attach_rate_list_v1)[::-1]
    floder_format = 'strategy_PGD-filterOnce-OppoPSel_v1-fixedEPS_{}-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_{}'
    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt(dir_root, floder_format,
                                                                                      fixedEPS_list,
                                                                                      np.array(attach_rate_list_v1)[
                                                                                      ::-1])
    # attach_rate_list_v1 = np.array(attach_rate_list_v1) + 0.1
    # [attach_rate, fixedEPS]
    ASR_v1 = (1 - mAPs * 100 / clean_mAP) * 100
    Permutations_v1 = P_Chamfer

    #### opopsite v2
    fixedEPS_list = [0.3]
    attach_rate_list_v2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    floder_format = 'strategy_PGD-filterOnce-OppoPSel_v2-fixedEPS_{}-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_{}'
    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt(dir_root, floder_format,
                                                                                      fixedEPS_list,
                                                                                      np.array(attach_rate_list_v2))
    # [attach_rate, fixedEPS]
    ASR_v2 = (1 - mAPs * 100 / clean_mAP) * 100
    Permutations_v2 = P_Chamfer

    #### Our method
    floder_format = 'strategy_PGD-filterOnce-PCSel-fixedEPS_{}-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_{}'
    fixedEPS_list = [0.3]
    our_attach_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt(dir_root, floder_format,
                                                                                      fixedEPS_list,
                                                                                      our_attach_rate_list)
    # [attach_rate, fixedEPS]
    our_ASR = (1 - mAPs * 100 / clean_mAP) * 100
    our_Permutations = P_Chamfer

    pylab.rcParams.update(myparams)  # 更新自己的设置

    linestyle = ['--', '--', '--', '--', '--']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    num_data = 10
    ax1.plot(np.hstack([attach_rate_list_v1]), np.hstack([0, ASR_v1[1:].reshape(-1)]),
             label="opposite_v1".format(fixedEPS_list[0]), marker='o',
             markersize=5)
    # ax1.plot(attach_rate_list_v11, ASR_v11[:, 0], label="opposite_v1.1".format(fixedEPS_list[0]), marker='s',
    #          markersize=5)
    ax1.plot(np.hstack([0, attach_rate_list_v2]), np.hstack([0, ASR_v2[-num_data:].reshape(-1)]),
             label="opposite_v2".format(fixedEPS_list[0]), marker='o',
             markersize=5)
    ax1.plot(np.hstack([0, our_attach_rate_list]), np.hstack([0, our_ASR[:, 0].reshape(-1)]),
             label="our method".format(fixedEPS_list[0]), marker='v',
             markersize=5)
    # ax1.legend(loc="upper left", title='mAP')
    ax1.set_ylabel('mAP drop (%)')
    ax1.set_xlabel('Modification Rate for Iteration')
    ax1.set_yticks(range(0, 100, 20))

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(np.hstack([attach_rate_list_v1]), np.hstack([Permutations_v1[:].reshape(-1)]),
             label="opposite_v1".format(fixedEPS_list[0]), marker='o',
             markersize=5, linestyle=linestyle[0])
    # ax2.plot(attach_rate_list_v11, Permutations_v11[:, 0], label="opposite_v1.1".format(fixedEPS_list[0]), marker='s',
    #          markersize=5, linestyle=linestyle[0])
    ax2.plot(np.hstack([0, attach_rate_list_v2]), np.hstack([0, Permutations_v2[-num_data:].reshape(-1)]),
             label="opposite_v2".format(fixedEPS_list[0]), marker='o',
             markersize=5, linestyle=linestyle[0])
    ax2.plot(np.hstack([0, our_attach_rate_list]), np.hstack([0, our_Permutations[:, 0].reshape(-1)]),
             label="Our method".format(fixedEPS_list[0]), marker='v',
             markersize=5, linestyle=linestyle[0])
    # ax2.legend(loc="lower right", title='Permutaion')
    ax2.set_ylabel('Perturbation')
    ax2.set_yticks(np.linspace(0, 0.6, 7))

    plt.savefig('./save_figures/opposite_v1-nuscenes_pillar.png', dpi=dpi,
                bbox_inches='tight')  # bbox_inches='tight'会裁掉多余的白边
    plt.show()


def curve_PCSel_opposite_v2_nuscenes_voxel():
    dir_root = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize/AdaptiveEPS/'

    clean_mAP = 65.3

    #### opopsite v1.1
    fixedEPS_list = [0.3]
    attach_rate_list_v11 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) + 0.2
    # [attach_rate, fixedEPS]
    mAPs = np.array([6.5, 5.9, 7.3, 9.5, 13.6, 17.2, 21.2, 23.7, 26.3]).reshape(-1, 1) / 100
    P_Chamfer = np.array([0.097, 0.121, 0.132, 0.139, 0.143, 0.143, 0.146, 0.145, 0.145]).reshape(-1, 1)
    ASR_v11 = (1 - mAPs * 100 / clean_mAP) * 100
    Permutations_v11 = P_Chamfer

    #### opopsite v2
    fixedEPS_list = [0.3]
    attach_rate_list_v2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    floder_format = 'strategy_PGD-filterOnce-OppoPSel_v2-fixedEPS_{}-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_{}'
    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt(dir_root, floder_format,
                                                                                      fixedEPS_list,
                                                                                      np.array(attach_rate_list_v2))
    # [attach_rate, fixedEPS]
    ASR_v2 = (1 - mAPs * 100 / clean_mAP) * 100
    Permutations_v2 = P_Chamfer

    pylab.rcParams.update(myparams)  # 更新自己的设置

    linestyle = ['--', '--', '--', '--', '--']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    num_data = 10

    ax1.plot(attach_rate_list_v11, ASR_v11[:, 0], label="opposite_v1.1".format(fixedEPS_list[0]), marker='s',
             markersize=5)
    ax1.plot(attach_rate_list_v2, ASR_v2[-num_data:], label="opposite_v2".format(fixedEPS_list[0]), marker='o',
             markersize=5)

    ax1.legend(loc="upper left", title='mAP')
    ax1.set_ylabel('mAP drop (%)')
    ax1.set_xlabel('Modification Rate for Iteration')
    ax1.set_yticks(range(0, 100, 20))

    ax2 = ax1.twinx()  # this is the important function

    ax2.plot(attach_rate_list_v11, Permutations_v11[:, 0], label="opposite_v1.1".format(fixedEPS_list[0]), marker='s',
             markersize=5, linestyle=linestyle[0])
    ax2.plot(attach_rate_list_v2, Permutations_v2[-num_data:], label="opposite_v2".format(fixedEPS_list[0]), marker='o',
             markersize=5, linestyle=linestyle[0])

    ax2.legend(loc="lower right", title='Permutaion')
    ax2.set_ylabel('Perturbation')
    ax2.set_yticks(np.linspace(0, 0.6, 7))

    plt.savefig('./save_figures/opposite_v2.png', dpi=dpi, bbox_inches='tight')  # bbox_inches='tight'会裁掉多余的白边
    plt.show()


def curve_PCSel_opposite_v1_waymo_voxel():
    dir_root = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV/AdaptiveEPS/'

    clean_mAP = 74.9
    clean_mAP = 67.4

    #### opopsite v1
    fixedEPS_list = [0.3]
    attach_rate_list_v1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # attach_rate_list_v1 = np.array(attach_rate_list_v1)[::-1]
    floder_format = 'strategy_PGD-filterOnce-OppoPSel_v1-fixedEPS_{}-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_{}'
    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt_waymo(dir_root, floder_format,
                                                                                            fixedEPS_list, np.array(
            attach_rate_list_v1)[::-1])
    # attach_rate_list_v1 = np.array(attach_rate_list_v1) + 0.1
    # [attach_rate, fixedEPS]
    ASR_v1 = (1 - mAPs / clean_mAP) * 100
    Permutations_v1 = P_Chamfer

    #### opopsite v2
    fixedEPS_list = [0.3]
    attach_rate_list_v2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    floder_format = 'strategy_PGD-filterOnce-OppoPSel_v2-fixedEPS_{}-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_{}'
    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt_waymo(dir_root, floder_format,
                                                                                            fixedEPS_list, np.array(
            attach_rate_list_v2))
    # [attach_rate, fixedEPS]
    ASR_v2 = (1 - mAPs / clean_mAP) * 100
    Permutations_v2 = P_Chamfer

    #### Our method
    floder_format = 'strategy_PGD-filterOnce-PCSel_323-fixedEPS_{}-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_{}'
    fixedEPS_list = [0.3]
    our_attach_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt_waymo(dir_root, floder_format,
                                                                                            fixedEPS_list,
                                                                                            our_attach_rate_list)
    # [attach_rate, fixedEPS]
    our_ASR = (1 - mAPs / clean_mAP) * 100
    our_Permutations = P_Chamfer

    pylab.rcParams.update(myparams)  # 更新自己的设置

    linestyle = ['--', '--', '--', '--', '--']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    num_data = 10
    ax1.plot(np.hstack([attach_rate_list_v1]), np.hstack([0, ASR_v1[1:].reshape(-1)]),
             label="opposite_v1".format(fixedEPS_list[0]), marker='o',
             markersize=5)
    # ax1.plot(attach_rate_list_v11, ASR_v11[:, 0], label="opposite_v1.1".format(fixedEPS_list[0]), marker='s',
    #          markersize=5)
    ax1.plot(np.hstack([0, attach_rate_list_v2]), np.hstack([0, ASR_v2[:].reshape(-1)]),
             label="opposite_v2".format(fixedEPS_list[0]), marker='o',
             markersize=5)
    ax1.plot(np.hstack([0, our_attach_rate_list]), np.hstack([0, our_ASR[:, 0].reshape(-1)]),
             label="our method".format(fixedEPS_list[0]), marker='v',
             markersize=5)
    # ax1.legend(loc="upper left", title='mAP')
    ax1.set_ylabel('mAP drop (%)')
    ax1.set_xlabel('Modification Rate for Iteration')
    ax1.set_yticks(range(0, 100, 20))

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(np.hstack([attach_rate_list_v1]), np.hstack([Permutations_v1[:].reshape(-1)]),
             label="opposite_v1".format(fixedEPS_list[0]), marker='o',
             markersize=5, linestyle=linestyle[0])
    # ax2.plot(attach_rate_list_v11, Permutations_v11[:, 0], label="opposite_v1.1".format(fixedEPS_list[0]), marker='s',
    #          markersize=5, linestyle=linestyle[0])
    ax2.plot(np.hstack([0, attach_rate_list_v2]), np.hstack([0, Permutations_v2[:].reshape(-1)]),
             label="opposite_v2".format(fixedEPS_list[0]), marker='o',
             markersize=5, linestyle=linestyle[0])
    ax2.plot(np.hstack([0, our_attach_rate_list]), np.hstack([0, our_Permutations[:, 0].reshape(-1)]),
             label="Our method".format(fixedEPS_list[0]), marker='v',
             markersize=5, linestyle=linestyle[0])
    # ax2.legend(loc="lower right", title='Permutaion')
    ax2.set_ylabel('Perturbation')
    ax2.set_yticks(np.linspace(0, 0.6, 7))

    plt.savefig('./save_figures/opposite_v1-waymo_voxel.png', dpi=dpi,
                bbox_inches='tight')  # bbox_inches='tight'会裁掉多余的白边
    plt.show()


def curve_PCSel_opposite_v1_waymo_pillar():
    dir_root = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV-pillar/AdaptiveEPS/'

    clean_mAP = 74.9
    clean_mAP = 67.4

    #### opopsite v1
    fixedEPS_list = [0.1]
    attach_rate_list_v1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # attach_rate_list_v1 = np.array(attach_rate_list_v1)[::-1]
    floder_format = 'strategy_PGD-filterOnce-OppoPSel_v1-fixedEPS_{}-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_{}'
    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt_waymo(dir_root, floder_format,
                                                                                            fixedEPS_list, np.array(
            attach_rate_list_v1)[::-1])
    # attach_rate_list_v1 = np.array(attach_rate_list_v1) + 0.1
    # [attach_rate, fixedEPS]
    ASR_v1 = (1 - mAPs / clean_mAP) * 100
    Permutations_v1 = P_Chamfer

    #### opopsite v2
    attach_rate_list_v2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    floder_format = 'strategy_PGD-filterOnce-OppoPSel_v2-fixedEPS_{}-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_{}'
    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt_waymo(dir_root, floder_format,
                                                                                            fixedEPS_list, np.array(
            attach_rate_list_v2))
    # [attach_rate, fixedEPS]
    ASR_v2 = (1 - mAPs / clean_mAP) * 100
    Permutations_v2 = P_Chamfer

    #### Our method
    floder_format = 'strategy_PGD-filterOnce-PCSel_323-fixedEPS_{}-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_{}'
    our_attach_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt_waymo(dir_root, floder_format,
                                                                                            fixedEPS_list,
                                                                                            our_attach_rate_list)
    # [attach_rate, fixedEPS]
    our_ASR = (1 - mAPs / clean_mAP) * 100
    our_Permutations = P_Chamfer

    pylab.rcParams.update(myparams)  # 更新自己的设置

    linestyle = ['--', '--', '--', '--', '--']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    num_data = 10
    ax1.plot(np.hstack([attach_rate_list_v1]), np.hstack([0, ASR_v1[1:].reshape(-1)]),
             label="opposite_v1".format(fixedEPS_list[0]), marker='o',
             markersize=5)
    # ax1.plot(attach_rate_list_v11, ASR_v11[:, 0], label="opposite_v1.1".format(fixedEPS_list[0]), marker='s',
    #          markersize=5)
    ax1.plot(np.hstack([0, attach_rate_list_v2]), np.hstack([0, ASR_v2[:].reshape(-1)]),
             label="opposite_v2".format(fixedEPS_list[0]), marker='o',
             markersize=5)
    ax1.plot(np.hstack([0, our_attach_rate_list]), np.hstack([0, our_ASR[:, 0].reshape(-1)]),
             label="our method".format(fixedEPS_list[0]), marker='v',
             markersize=5)
    # ax1.legend(loc="upper left", title='mAP')
    ax1.set_ylabel('mAP drop (%)')
    ax1.set_xlabel('Modification Rate for Iteration')
    ax1.set_yticks(range(0, 100, 20))

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(np.hstack([attach_rate_list_v1]), np.hstack([Permutations_v1[:].reshape(-1)]),
             label="opposite_v1".format(fixedEPS_list[0]), marker='o',
             markersize=5, linestyle=linestyle[0])
    # ax2.plot(attach_rate_list_v11, Permutations_v11[:, 0], label="opposite_v1.1".format(fixedEPS_list[0]), marker='s',
    #          markersize=5, linestyle=linestyle[0])
    ax2.plot(np.hstack([0, attach_rate_list_v2]), np.hstack([0, Permutations_v2[:].reshape(-1)]),
             label="opposite_v2".format(fixedEPS_list[0]), marker='o',
             markersize=5, linestyle=linestyle[0])
    ax2.plot(np.hstack([0, our_attach_rate_list]), np.hstack([0, our_Permutations[:, 0].reshape(-1)]),
             label="Our method".format(fixedEPS_list[0]), marker='v',
             markersize=5, linestyle=linestyle[0])
    # ax2.legend(loc="lower right", title='Permutaion')
    ax2.set_ylabel('Perturbation')
    ax2.set_yticks(np.linspace(0, 0.6, 7))

    plt.savefig('./save_figures/opposite_v1-waymo_pillar.png', dpi=dpi,
                bbox_inches='tight')  # bbox_inches='tight'会裁掉多余的白边
    plt.show()


if __name__ == '__main__':
    # plot_doubleY()

    # curve_PCSel_nuscenes_voxel()
    # curve_PCSel_nuscenes_pillar()
    # curve_PCSel_waymo_voxel()
    # curve_PCSel_waymo_pillar()

    curve_PCSel_opposite_v1_nuscenes_voxel()
    # curve_PCSel_opposite_v1_nuscenes_pillar()
    # curve_PCSel_opposite_v1_waymo_voxel()
    # curve_PCSel_opposite_v1_waymo_pillar()
