import numpy as np
# import open3d as o3d
# import openmesh as om
from openmesh import *
import os
from lidar_simulation.lidar import Lidar
import demo_GA_Attack
import torch
import trimesh
import sys
import datetime
from torch.utils.tensorboard import SummaryWriter

# 遗传算法参数
DNA_size = None  # DNA size
CROSS_RATE = 0.1
MUTATE_RATE = 0.2
MUTATE_STD = 0.05
POP_SIZE = 160
N_GENERATIONS = 200 # debug
LEFTOVER_RATE = 0.5
# 高斯噪声参数
GN_mean = 0.0
GN_std = 0.02

# 对抗攻击
Epsilon = 0.02 # 2厘米
# Epsilon = float('inf')
Target_Label = 0  # ['Car', 'Pedestrian', 'Cyclist']
Top_M = 100
EXP_NAME = 'fixCarPosi-RenderPosi'


save_dir_root = r'/data/dataset_wujunqi/GeneticAlgorithm/GeneticAlgorithm-v2-{}'\
    .format(EXP_NAME)
os.makedirs(save_dir_root, exist_ok=True)
mesh_path = r'../data/GeneticAlgorithm/cone_0001.ply'
target_mesh_path = r'../data/GeneticAlgorithm/car_0004.off'
PC_background_path = r'../data/GeneticAlgorithm/lidar_background-000306.bin' # 背景点云

max_float=float('inf')
min_float = 0.0000001

# 对抗物体调整
off_set_list = [7, 0, -1.73]
POSITION = (0, 0, 0)
scalar_rate = 1.0/15.875


def loadPCL(PCL, flag=True):
    if flag:
        PCL = np.fromfile(PCL, dtype=np.float32)
        PCL = PCL.reshape((-1, 4))
    else:
        PCL = pypcd.PointCloud.from_path(PCL)
        PCL = np.array(tuple(PCL.pc_data.tolist()))
        PCL = np.delete(PCL, -1, axis=1)
    return PCL

# f-function in the paper
def CW_f(outputs, labels, is_targeted=True, kappa=0):
    # outputs: [100, 1]
    values, indices = torch.topk(input=outputs, k=2, dim=0, largest=True, sorted=True)
    logits_1st = outputs[indices[0]]
    logits_2nd = outputs[indices[1]]

    # if is_targeted:
    #     return torch.clamp((i - j), min=-kappa)
    # else:
    #     return torch.clamp((j - i), min=-kappa)

    # one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)
    outputs_copy = outputs.clone().detach()
    one_hot_labels = torch.eye(len(outputs_copy[:, 0]))[labels]

    i, _ = torch.max((1 - one_hot_labels) * outputs_copy[:, 0], dim=0)  # get the second largest logit
    j = torch.masked_select(outputs[:, 0], one_hot_labels.bool())  # get the largest logit

    if is_targeted:
        return torch.clamp((i - j), min=-kappa)
    else:
        return torch.clamp((j - i), min=-kappa)


def L2_mesh(mesh1, mesh2):
    point_array1 = mesh1.vertices
    point_array2 = mesh2.vertices
    dist = (point_array1 - point_array2).reshape(-1)
    l2_dist = torch.norm(torch.tensor(dist), p=2, dim=0)
    return l2_dist.numpy()

def show_width_height(vertices):
    x_max = np.max(vertices[:, 0])
    x_min = np.min(vertices[:, 0])
    dist_x = x_max - x_min
    print("x_min={:.3f}, x_max={:.3f}, dist_x={:.3f}".format(x_min, x_max, dist_x))
    y_max = np.max(vertices[:, 1])
    y_min = np.min(vertices[:, 1])
    dist_y = y_max - y_min
    print("y_min={:.3f}, y_max={:.3f}, dist_y={:.3f}".format(y_min, y_max, dist_y))
    z_max = np.max(vertices[:, 2])
    z_min = np.min(vertices[:, 2])
    dist_z = z_max - z_min
    print("z_min={:.3f}, z_max={:.3f}, dist_z={:.3f}".format(z_min, z_max, dist_z))


class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, mesh, off_set_list, scalar_rate=0.1, leftover_rate=0.5, logger=None):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.mesh_origin = mesh
        self.leftover_rate = leftover_rate
        self.logger = logger
        # tensorboard_dir = './GeneticAlgorithm/log_tensorboard/' + EXP_NAME
        # os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter('./GeneticAlgorithm/log_tensorboard/' + EXP_NAME)

        # 预处理 -----------------
        vertices = np.array(mesh.vertices)
        print('原始位置：')
        show_width_height(vertices)

        vertices *= scalar_rate
        print("缩放后：")
        show_width_height(vertices)

        for i in range(len(off_set_list)):
            vertices[:, i] += off_set_list[i]
        print("加偏移量 {} 后：".format(off_set_list))
        show_width_height(vertices)
        # 预处理 -----------------
        self.save_mesh(vertices, os.path.join(r'../data/GeneticAlgorithm', 'origin_location.ply'))
        vertices = np.expand_dims(vertices, axis=0)
        assert len(vertices.shape) == 3
        self.pop = np.repeat(vertices, repeats=pop_size, axis=0) # 复制
        # 添加高斯噪声
        gaussian_noise = np.random.normal(loc=GN_mean, scale=GN_std, size=self.pop.shape)
        pop_adv = self.pop + gaussian_noise
        delta = np.clip(pop_adv - self.pop, a_min=-Epsilon, a_max=Epsilon)
        self.pop = self.pop + delta
        # 得到被攻击的目标类别的特征
        self.get_target_object_features()


    def get_target_object_features(self):
        mesh = trimesh.load(target_mesh_path)
        # 打印网格信息
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        self.logger.info("目标类别 点的维度: {}".format(vertices.shape))
        self.logger.info("目标类别 面的维度: {}".format(faces.shape))
        # 加偏置
        for i in range(len(off_set_list)):
            vertices[:, i] += off_set_list[i]
        # lidar渲染
        point_cloud_background = loadPCL(PC_background_path, True)  # mean:0.2641031, std:0.12991029
        intensity_mean = np.mean(point_cloud_background[:, 3])

        point_cloud_object_render = Lidar(delta_azimuth=2 * np.pi / 2000,
                                          delta_elevation=np.pi / 800,
                                          position=POSITION).sample_3d_model_gpu(vertices, faces)  # 确认渲染器使用
        intensity_array = np.ones((point_cloud_object_render.shape[0], 1), dtype=np.float32) * intensity_mean
        point_cloud_input = np.concatenate((point_cloud_object_render, intensity_array), axis=1)
        bin_file = target_mesh_path.replace(".off", ".bin")
        point_cloud_input = point_cloud_input.astype(np.float32)
        point_cloud_input.tofile(bin_file)

        # 放在背景里方便可视化
        point_cloud_scense  = np.concatenate((point_cloud_input, point_cloud_background), axis=0)  # 加可视化
        bin_file_scense = target_mesh_path.replace('.off', '-scense.bin')
        point_cloud_scense = point_cloud_scense.astype(np.float32)
        point_cloud_scense.tofile(bin_file_scense)

        # self.model = demo_GA_Attack.setup(bin_file, self.model)
        _, batch_dict_list = demo_GA_Attack.network_forward(self.logger, bin_file)
        # print("打印键值ga")
        # print(batch_dict_list[0])
        # for key in batch_dict_list[0]:
        #     print(key) # 打印key
        self.target_features = batch_dict_list[0]['point_features']
        self.target_rpn_roi = batch_dict_list[0]['rois']
        self.target_rcnn_roi = batch_dict_list[0]['batch_box_preds']


    def save_mesh(self, vertices, save_path):
        faces = self.mesh_origin.faces
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        # save
        result = trimesh.exchange.ply.export_ply(mesh, encoding='ascii')
        output_file = open(save_path, "wb+")
        output_file.write(result)
        output_file.close()


    def translate_mesh(self, mesh_list):
        faces = np.array(self.mesh_origin.faces)
        vertices = np.zeros((self.DNA_size, 3), dtype=np.float32)
        for p in range(self.pop_size):
            for point in range(self.DNA_size):
                [x, y, z] = self.pop[p, point, :]
                vertices[point, :] = [x, y, z]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh_list.append(mesh)

    def get_fitness(self, generation):
        '''
        1.使用点和面的关系构建对抗物体mesh
        2.对抗物体mesh输入到Lidar渲染器得到对抗物体点云
        3.（对抗物体点云 + 背景点云)输入PointRCNN得到预测结果
        4.根据预测结果计算fitness
        '''
        # 1
        mesh_list = []
        self.translate_mesh(mesh_list)
        # 2
        point_cloud_background = loadPCL(PC_background_path, True) # mean:0.2641031, std:0.12991029
        intensity_mean = np.mean(point_cloud_background[:, 3])
        # 3
        # 生成点云并保存
        save_dir = os.path.join(save_dir_root, 'iter' + str(generation))
        os.makedirs(save_dir, exist_ok=True)
        for p in range(self.pop_size):
            vertices = mesh_list[p].vertices
            polygons = mesh_list[p].faces
            point_cloud_object_render = Lidar(delta_azimuth=2 * np.pi / 2000,
                                delta_elevation=np.pi / 800,
                                position=POSITION).sample_3d_model_gpu(vertices, polygons) # 确认渲染器使用
            intensity_array = np.ones((point_cloud_object_render.shape[0], 1), dtype=np.float32) * intensity_mean
            point_cloud_object = np.concatenate((point_cloud_object_render, intensity_array), axis=1)
            # point_cloud_input = np.concatenate((point_cloud_object, point_cloud_background), axis=0)  # 加可视化

            point_cloud_input = point_cloud_object
            bin_file = os.path.join(save_dir, 'iter_' + str(generation) + '-popID_' + str(p) + '.bin')
            point_cloud_input = point_cloud_input.astype(np.float32)
            point_cloud_input.tofile(bin_file)

        # 点云输入网络
        # self.model = demo_GA_Attack.setup(save_dir, self.model)
        pred_dicts_list, batch_dict_list = demo_GA_Attack.network_forward(self.logger, save_dir)

        fitness_list = []
        loss_cls_list = []
        loss_features_list = []
        loss_box_list = []
        loss_L2_list = []
        for p in range(self.pop_size):
            # Loss_cls
            Z_bg = batch_dict_list[p]['batch_cls_preds'][0, :, 0] # [1, 100, 1]. rcnn背景分数 ?
            Z_rpn = batch_dict_list[p]['roi_scores'][0] # [1, 100]. rpn oojectiveness scores
            Z_t = batch_dict_list[p]['roi_cls_logit'][0, :, Target_Label] # [1, 100, 3]   分类成目标类别的logits
            k_threshold = 0.9
            loss_cls = ((Z_t[Z_t < k_threshold]) * (Z_bg - Z_rpn - Z_t)).sum()

            # loss_features
            adv_object_features = batch_dict_list[p]['point_features']
            loss_features = torch.norm(self.target_features - adv_object_features, p=2)

            # loss_box
            target_orientation = self.target_rpn_roi[0, 0, -1]
            adv_object_orientation = batch_dict_list[p]['rois'][0, 0, -1]
            loss_box =torch.norm(target_orientation - adv_object_orientation, p=2)

            # loss_realizability
            # loss_printability =

            alpha = 0.001 # 0.001
            beta = 0.1 # 0.001
            loss_total = loss_cls + alpha * loss_features + beta * loss_box
            loss_L2 = L2_mesh(self.mesh_origin, mesh_list[p])

            loss_cls_list.append(loss_cls.cpu().numpy())
            loss_features_list.append(loss_features.cpu().numpy())
            loss_box_list.append(loss_box.cpu().numpy())
            loss_L2_list.append(loss_L2)
            fitness_list.append(loss_total.cpu().numpy())

        loss_cls_array = np.array(loss_cls_list)
        loss_features_array = np.array(loss_features_list)
        loss_box_array = np.array(loss_box_list)
        loss_L2_array = np.array(loss_L2_list)
        fitness_array = np.array(fitness_list)

        self.logger.info('loss_cls: mean={:.3f}, std={:.3f}'.format(loss_cls_array.mean(), loss_cls_array.std()))
        self.logger.info('loss_features: mean={:.3f}, std={:.3f}'.format(loss_features_array.mean(), loss_features_array.std()))
        self.logger.info('loss_box: mean={:.3f}, std={:.3f}'.format(loss_box_array.mean(), loss_box_array.std()))
        self.logger.info('loss_L2: mean={:.3f}, std={:.3f}'.format(loss_L2_array.mean(), loss_L2_array.std()))
        self.logger.info('fitness: mean={:.3f}, std={:.3f}'.format(fitness_array.mean(), fitness_array.std()))

        self.writer.add_scalar('loss/loss_cls', loss_cls_array.mean(), generation)
        self.writer.add_scalar('loss/loss_features', loss_features_array.mean(), generation)
        self.writer.add_scalar('loss/loss_box', loss_box_array.mean(), generation)
        self.writer.add_scalar('loss/loss_L2', loss_L2_array.mean(), generation)
        self.writer.add_scalar('loss/fitness', fitness_array.mean(), generation)
        return fitness_list


    def select_probability(self, fitness):
        print('fitness: {}'.format(fitness))
        print("p={}".format((np.array(fitness) + min_float) / (np.array(fitness).sum() + EPS)))
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True,
                               p=(np.array(fitness) + min_float) / (np.array(fitness).sum() + min_float))
        return self.pop[idx]

    def select_leftover(self, fitness):
        left_over_size = int(self.pop_size * self.leftover_rate)
        index_sorted = np.argsort(fitness, axis=0) # 从小到大排序
        pop_best = self.pop[index_sorted[:left_over_size]] # 取前面的部分
        pop_leftover = self.pop[index_sorted[left_over_size:]] # 取后面的部分
        return pop_best, pop_leftover

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size * self.leftover_rate, size=1) # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool) # choose crossover points
            parent[cross_points] = pop[i_, cross_points]
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size): # 每一个点都单独变异
            if np.random.rand() < self.mutate_rate:
                gaussian_noise = np.random.normal(loc=GN_mean, scale=GN_std, size=3)
                child[point] += gaussian_noise
        return child

    def evolve(self, fitness_list):
        pop_best, pop_leftover = self.select_leftover(fitness_list)
        pop_leftover_copy = pop_leftover.copy()
        for parent in pop_leftover:  # for every parent
            child = self.crossover(parent, pop_leftover_copy)
            child = self.mutate(child)
            # Epsilon裁剪
            vertices = np.array(self.mesh_origin.vertices)
            delta = np.clip(child - vertices, a_min=-Epsilon, a_max=Epsilon)
            child = vertices + delta

            parent[:] = child
        self.pop = np.concatenate((pop_best, pop_leftover), axis=0)


if __name__ == '__main__':
    logger = demo_GA_Attack.initial(EXP_NAME)
    # mesh_path = r'../data/GeneticAlgorithm/chair-origin.ply'
    mesh = trimesh.load(mesh_path)
    # 打印网格信息
    v = mesh.vertices
    f = mesh.faces
    logger.info("点的维度: {}".format(v.shape))
    logger.info("面的维度: {}".format(f.shape))
    DNA_size = v.shape[0]
    ga = GA(DNA_size=DNA_size,
            cross_rate=CROSS_RATE,
            mutation_rate=MUTATE_RATE,
            pop_size=POP_SIZE,
            mesh=mesh,
            off_set_list=off_set_list,
            scalar_rate=scalar_rate,
            leftover_rate=LEFTOVER_RATE,
            logger=logger)

    for generation in range(N_GENERATIONS):
        logger.info(' -------------- iteration: {} -----------------'.format(generation))
        fitness_list = ga.get_fitness(generation)
        # fitness = np.random.rand(POP_SIZE)
        ga.evolve(fitness_list)
        best_idx = np.argmin(fitness_list)
        logger.info('Gen: {} | best fit: {:.2f} | best_idx: {}'.format(generation, fitness_list[best_idx], best_idx))

        if generation % 5 == 0:
            save_dir_best_fitness = os.path.join(r'../data/GeneticAlgorithm/Best_Fitness', EXP_NAME, 'iter_{}'.format(generation))
            os.makedirs(save_dir_best_fitness, exist_ok=True)
            ga.save_mesh(ga.pop[best_idx, :, :], os.path.join(save_dir_best_fitness, 'best_fitness-id_{}.ply'.format(best_idx)))
