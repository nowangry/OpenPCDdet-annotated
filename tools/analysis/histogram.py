import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def min_max_normalize(data):
    """
    最小-最大归一化
    :param data: 待归一化的数据，一维数组形式
    :return: 归一化后的数据
    """
    min_val = min(data)
    max_val = max(data)
    norm_data = [(x - min_val) / (max_val - min_val) for x in data]
    return np.array(norm_data)


def l2_normalize(data):
    """
    L2范数归一化
    :param data: 待归一化的数据，一维数组形式
    :return: 归一化后的数据
    """
    norm = np.linalg.norm(data)
    norm_data = [x / norm for x in data]
    return np.array(norm_data)


def z_score_normalize(data):
    """
    Z-score标准化
    :param data: 待归一化的数据，一维数组形式
    :return: 归一化后的数据
    """
    mean_val = np.mean(data)
    std_val = np.std(data)
    norm_data = [(x - mean_val) / std_val for x in data]
    return np.array(norm_data)


def MyHist(x_value=np.random.randint(140, 180, 200), bins=100, title_name="data analyze", save_path=''):
    """
    绘制直方图
    :return:
    """
    # 设置在jupyter中matplotlib的显示情况
    # % matplotlib inline

    # 解决中文显示问题
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.hist(x_value, bins=bins, edgecolor="r")

    plt.title(title_name)
    plt.xlabel("value")
    plt.ylabel("number")
    if not save_path == '':
        plt.savefig(save_path)

    plt.show()


if __name__ == "__main__":
    MyHist()
