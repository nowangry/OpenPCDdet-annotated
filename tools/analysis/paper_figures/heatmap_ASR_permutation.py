import numpy as np
import seaborn as sns
import os
import matplotlib
import matplotlib.pyplot as plt
from tools.analysis.paper_figures.util import *

def test_heatmap():
    '''

    https://blog.csdn.net/clksjx/article/details/104525411
    '''

    # rc = {'font.sans-serif': ['WenQuanYi Micro Hei', 'DejaVu Sans', 'Bitstream Vera Sans']}
    # sns.set(rc=rc)

    sns.set_theme(font='Times New Roman')

    # 设置西文字体为新罗马字体
    from matplotlib import rcParams
    # plt.rc('font', family='Times New Roman')
    # del matplotlib.font_manager.weight_dict['roman']
    # matplotlib.font_manager._rebuild()

    # # 字体
    # from matplotlib import rcParams
    # config = {
    #     "font.family": 'Times New Roman',  # 设置字体类型
    #     # "font.size": 80,
    #     #     "mathtext.fontset":'stix',
    # }
    # rcParams.update(config)

    data = np.random.rand(10, 12)
    f, ax = plt.subplots()
    ax = sns.heatmap(data, cmap='RdBu', ax=ax, vmin=0, vmax=1, annot=True, fmt='0.1g')

    # 设置坐标字体方向
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title('Heatmap for test')  # 图标题
    ax.set_xlabel('x label')  # x轴标题
    ax.set_ylabel('y label')

    plt.show()

    figure = ax.get_figure()
    figure.savefig('sns_heatmap.jpg')  # 保存图片


def test1():
    # coding=utf-8

    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    x = np.array([1, 2, 3, 4, 5, 6])
    VGG_supervised = np.array([2.9749694, 3.9357018, 4.7440844, 6.482254, 8.720203, 13.687582])
    VGG_unsupervised = np.array([2.1044724, 2.9757383, 3.7754183, 5.686206, 8.367847, 14.144531])
    ourNetwork = np.array([2.0205495, 2.6509762, 3.1876223, 4.380781, 6.004548, 9.9298])

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(10, 5))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    plt.plot(x, VGG_supervised, marker='o', color="blue", label="VGG-style Supervised Network", linewidth=1.5)
    plt.plot(x, VGG_unsupervised, marker='o', color="green", label="VGG-style Unsupervised Network", linewidth=1.5)
    plt.plot(x, ourNetwork, marker='o', color="red", label="ShuffleNet-style Network", linewidth=1.5)

    group_labels = ['Top 0-5%', 'Top 5-10%', 'Top 10-20%', 'Top 20-50%', 'Top 50-70%', ' Top 70-100%']  # x轴刻度的标识
    plt.xticks(x, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=12, fontweight='bold')
    # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
    plt.xlabel("Performance Percentile", fontsize=13, fontweight='bold')
    plt.ylabel("4pt-Homography RMSE", fontsize=13, fontweight='bold')
    plt.xlim(0.9, 6.1)  # 设置x轴的范围
    plt.ylim(1.5, 16)

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细

    plt.savefig('./filename.svg', format='svg')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    plt.show()


def heatmap(data, fmt='0.2f', xticklabels=False, yticklabels=False, x_label=None, y_label=None,
            title_name='figure.jpg'):
    # sns.set(font_scale=1.5)
    plt.rc('font', family='Times New Roman', size=18)

    f, ax = plt.subplots()
    ax = sns.heatmap(data=data,
                     xticklabels=xticklabels,
                     yticklabels=yticklabels,
                     cmap='RdBu_r',
                     ax=ax,
                     annot=True,
                     fmt=fmt,
                     )

    # 设置坐标字体方向
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
    ax.set_title(title_name)  # 图标题
    ax.set_xlabel(x_label)  # x轴标题
    ax.set_ylabel(y_label)
    plt.show()

    figure = ax.get_figure()
    figure.savefig(os.path.join("./save_figures", title_name), bbox_inches='tight')


def heatmap_PCSel():
    dir_root = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize/AdaptiveEPS/'
    floder_format = 'strategy_PGD-filterOnce-PCSel-fixedEPS_{}-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_{}'

    fixedEPS_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    attach_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt(dir_root, floder_format,
                                                                                      fixedEPS_list, attach_rate_list)

    heatmap(mAPs * 100,
            xticklabels=fixedEPS_list,
            yticklabels=['10%', '20%', '30%', '40%', '50%'],
            x_label='eps_iter',
            y_label='modication_rate_iter',
            title_name='mAPs',
            )
    ASR = 1 - mAPs * 100 / 65.3
    heatmap(ASR,
            xticklabels=fixedEPS_list,
            yticklabels=['10%', '20%', '30%', '40%', '50%'],
            x_label='eps_iter',
            y_label='modication_rate_iter',
            title_name='ASR',
            )
    heatmap(modification_rate,
            xticklabels=fixedEPS_list,
            yticklabels=['10%', '20%', '30%', '40%', '50%'],
            x_label='eps_iter',
            y_label='modication_rate_iter',
            title_name='modification_rate',
            )
    heatmap(P_L1,
            fmt='0.2g',
            xticklabels=fixedEPS_list,
            yticklabels=['10%', '20%', '30%', '40%', '50%'],
            x_label='eps_iter',
            y_label='modication_rate_iter',
            title_name='P_L1',
            )
    heatmap(P_L2,
            xticklabels=fixedEPS_list,
            yticklabels=['10%', '20%', '30%', '40%', '50%'],
            x_label='eps_iter',
            y_label='modication_rate_iter',
            title_name='P_L2',
            )
    heatmap(P_Chamfer,
            fmt='0.2g',
            xticklabels=fixedEPS_list,
            yticklabels=['10%', '20%', '30%', '40%', '50%'],
            x_label='eps_iter',
            y_label='modication_rate_iter',
            title_name='P_Chamfer',
            )
    heatmap(ASR / P_Chamfer,
            fmt='0.2g',
            xticklabels=fixedEPS_list,
            yticklabels=['10%', '20%', '30%', '40%', '50%'],
            x_label='eps_iter',
            y_label='modication_rate_iter',
            title_name='ASR div P_Chamfer',
            )
    heatmap(ASR / modification_rate,
            fmt='0.2g',
            xticklabels=fixedEPS_list,
            yticklabels=['10%', '20%', '30%', '40%', '50%'],
            x_label='eps_iter',
            y_label='modication_rate_iter',
            title_name='ASR div modification_rate',
            )


if __name__ == '__main__':
    # test1()
    # test_heatmap()
    heatmap_PCSel()
