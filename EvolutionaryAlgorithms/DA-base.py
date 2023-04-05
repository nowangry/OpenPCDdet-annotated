import numpy as np
# import open3d as o3d
# import openmesh as om
from openmesh import *
import os
from lidar_simulation.lidar import Lidar
import demo_EA_Attack
import torch
import trimesh
import sys
import datetime
from torch.utils.tensorboard import SummaryWriter

# 实验配置文件 ----------------------------------------------
YAML_file_path = r'cfgs/bin/DA-exp_1.4.1.yaml'
# YAML_file_path = r'./cfgs/DA-debug.yaml'


cfg = demo_EA_Attack.load_config(YAML_file_path)
logger_path = os.path.join('./log/', cfg.EXP_NAME, 'log_txt', 'log_DA_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
logger = demo_EA_Attack.load_logger(logger_path)
logger.info('YAML_file_path: {}'.format(YAML_file_path))
# 路径编辑 -------------------------------------------------
save_dir_root = os.path.join(cfg.PATH.result_output_dir_root, cfg.EXP_NAME)
os.makedirs(save_dir_root, exist_ok=True)
tensorboard_path = os.path.join('./log/', cfg.EXP_NAME, 'log_tensorboard/')
save_dir_train_root = os.path.join(save_dir_root, 'train')
save_dir_best_fitness_root = os.path.join(save_dir_root, 'best_fitness')

off_set_list = cfg.ADV_OBJECT.object_position
target_mesh_path = cfg.PATH.target_mesh_path
PC_background_path = cfg.PATH.PC_background_path
lidar_position = tuple(cfg.ADV_OBJECT.lidar_position)
# --------------------------------------------------------
max_float=float('inf')
min_float = 0.0000001

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
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, mesh,
                 scalar_rate=0.1, leftover_rate=0.5, logger=None):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.leftover_rate = leftover_rate
        self.logger = logger
        # tensorboard_dir = './GeneticAlgorithm/log_tensorboard/' + EXP_NAME
        # os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_path)
        self.car_pro_mean_list = []
        self.best_fitness_car_pro_list = []
        self.point_cloud_background = loadPCL(PC_background_path, True)  # mean:0.2641031, std:0.12991029
        self.intensity_mean = np.mean(self.point_cloud_background[:, 3])


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
        self.object_location_path = os.path.join(r'./data/outputs', cfg.EXP_NAME, 'object_location.ply')
        self.mesh_object = self.save_mesh(vertices, mesh.faces, self.object_location_path)

        vertices = np.expand_dims(vertices, axis=0)
        assert len(vertices.shape) == 3
        self.pop = np.repeat(vertices, repeats=pop_size, axis=0) # 复制
        # 添加高斯噪声
        gaussian_noise = np.random.normal(loc=cfg.ADV_PARAMETERS.GN_mean, scale=cfg.ADV_PARAMETERS.GN_std, size=self.pop.shape)
        pop_adv = self.pop + gaussian_noise
        delta = np.clip(pop_adv - self.pop, a_min=-cfg.ADV_PARAMETERS.Epsilon, a_max=cfg.ADV_PARAMETERS.Epsilon)
        self.pop = self.pop + delta
        # 得到被攻击的目标类别的特征
        self.get_target_object_features()


    def get_target_object_features(self):
        self.logger.info(" 对抗物体：mesh --> 点云")
        self.rendering_and_save(self.object_location_path, False)
        self.logger.info(" 目标类别：mesh --> 点云")
        bin_file = self.rendering_and_save(target_mesh_path, True)

        _, batch_dict_list, _ = demo_EA_Attack.network_forward(self.logger, cfg, bin_file)
        # print("打印键值ga")S
        # print(batch_dict_list[0])
        # for key in batch_dict_list[0]:
        #     print(key) # 打印key
        self.target_features = batch_dict_list[0]['point_features']
        self.target_rpn_roi = batch_dict_list[0]['rois']
        self.target_rcnn_roi = batch_dict_list[0]['batch_box_preds']


    def rendering_and_save(self, mesh_path, is_off_set):
        mesh = trimesh.load(mesh_path)
        # 打印网格信息
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        self.logger.info("---- 点的维度: {}".format(vertices.shape))
        self.logger.info("---- 面的维度: {}".format(faces.shape))
        if is_off_set:
            # 加偏置
            for i in range(len(off_set_list)):
                vertices[:, i] += off_set_list[i]

        # lidar渲染
        point_cloud_object_render = Lidar(delta_azimuth=2 * np.pi / 2000,
                                          delta_elevation=np.pi / 800,
                                          position=lidar_position).sample_3d_model_gpu(vertices, faces)  # 确认渲染器使用
        intensity_array = np.ones((point_cloud_object_render.shape[0], 1), dtype=np.float32) * self.intensity_mean
        point_cloud_input = np.concatenate((point_cloud_object_render, intensity_array), axis=1)

        # 单独保存物体
        bin_file = mesh_path.replace(os.path.splitext(mesh_path)[-1], ".bin").replace("inputs", "outputs/" + cfg.EXP_NAME)
        point_cloud_input = point_cloud_input.astype(np.float32)
        point_cloud_input.tofile(bin_file)
        self.logger.info("通过lidar模拟器后的点云维度：{}".format(point_cloud_input.shape))

        # 物体放在背景里方便可视化
        point_cloud_scense = np.concatenate((point_cloud_input, self.point_cloud_background), axis=0)  # 加可视化
        bin_file_scense = mesh_path.replace(os.path.splitext(mesh_path)[-1], '-scense.bin').replace("inputs", "outputs/" + cfg.EXP_NAME)
        point_cloud_scense = point_cloud_scense.astype(np.float32)
        point_cloud_scense.tofile(bin_file_scense)

        return bin_file


    def save_mesh(self, vertices, faces, save_path):
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        (filedir, filename) = os.path.split(save_path)
        os.makedirs(filedir, exist_ok=True)
        # save
        result = trimesh.exchange.ply.export_ply(mesh, encoding='ascii')
        output_file = open(save_path, "wb+")
        output_file.write(result)
        output_file.close()
        return mesh


    def translate_mesh(self, mesh_list):
        faces = np.array(self.mesh_object.faces)
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
        # point_cloud_background = loadPCL(PC_background_path, True) # mean:0.2641031, std:0.12991029
        # intensity_mean = np.mean(point_cloud_background[:, 3])
        # 3
        # 生成点云并保存
        # is_flag = True
        save_dir = os.path.join(save_dir_root, 'Generate', 'iter' + str(generation))
        os.makedirs(save_dir, exist_ok=True)
        for p in range(self.pop_size):
            vertices = mesh_list[p].vertices
            polygons = mesh_list[p].faces
            point_cloud_object_render = Lidar(delta_azimuth=2 * np.pi / 2000,
                                delta_elevation=np.pi / 800,
                                position=lidar_position).sample_3d_model_gpu(vertices, polygons) # 确认渲染器使用
            intensity_array = np.ones((point_cloud_object_render.shape[0], 1), dtype=np.float32) * self.intensity_mean
            point_cloud_object = np.concatenate((point_cloud_object_render, intensity_array), axis=1)
            # point_cloud_input = np.concatenate((point_cloud_object, self.point_cloud_background), axis=0)  # 加可视化

            point_cloud_input = point_cloud_object
            bin_file = os.path.join(save_dir, 'iter_' + str(generation) + '-popID_' + str(p) + '.bin')
            point_cloud_input = point_cloud_input.astype(np.float32)
            point_cloud_input.tofile(bin_file)

            # if is_flag:
            #     is_flag = False
        self.logger.info("====> 对抗物体维度 point_cloud_input.shape: {}".format(point_cloud_input.shape))

        # 点云输入网络
        # self.model = demo_GA_Attack.setup(save_dir, self.model)
        pred_dicts_list, batch_dict_list, car_probability_list = demo_EA_Attack.network_forward(self.logger, cfg, save_dir)

        car_pro_mean = np.array(car_probability_list).mean()
        car_pro_std = np.array(car_probability_list).std()
        self.car_pro_mean_list.append(car_pro_mean)
        logger.info('iter: {}. 分类为车的平均概率为： {:.3f}%. STD为：{:.3f}'.format(generation, car_pro_mean * 100, car_pro_std))
        self.writer.add_scalar('car_probability/car_p_mean', car_pro_mean, generation)
        self.writer.add_scalar('car_probability/car_p_std', car_pro_mean, generation)

        fitness_list = []
        loss_cls_list = []
        loss_features_list = []
        loss_box_list = []
        loss_L2_list = []
        for p in range(self.pop_size):
            # Loss_cls
            Z_bg = batch_dict_list[p]['batch_cls_preds'][0, :, 0] # [1, 100, 1]. rcnn背景分数 ?
            Z_rpn = batch_dict_list[p]['roi_scores'][0] # [1, 100]. rpn oojectiveness scores
            Z_t = batch_dict_list[p]['roi_cls_logit'][0, :, cfg.ADV_PARAMETERS.Target_Label] # [1, 100, 3]   分类成目标类别的logits
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
            loss_features *= cfg.ADV_PARAMETERS.weight_feat
            loss_box *= cfg.ADV_PARAMETERS.weight_box

            loss_total = loss_cls + loss_features + loss_box
            loss_L2 = L2_mesh(self.mesh_object, mesh_list[p])

            loss_cls_list.append(loss_cls.cpu().numpy())
            loss_features_list.append(loss_features.cpu().numpy())
            loss_box_list.append(loss_box.cpu().numpy())
            loss_L2_list.append(loss_L2)
            fitness_list.append(loss_total.cpu().numpy())

        # 记录数据
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

        best_idx = np.argmin(fitness_list)
        best_fitness_car_pro = car_probability_list[best_idx]
        self.best_fitness_car_pro_list.append(best_fitness_car_pro)
        self.writer.add_scalar('car_probability/best_fitness', best_fitness_car_pro, generation)
        return fitness_list


    # def select_probability(self, fitness):
    #     print('fitness: {}'.format(fitness))
    #     print("p={}".format((np.array(fitness) + min_float) / (np.array(fitness).sum() + EPS)))
    #     idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True,
    #                            p=(np.array(fitness) + min_float) / (np.array(fitness).sum() + min_float))
    #     return self.pop[idx]

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
            parent[cross_points] = pop[i_, cross_points, :]
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size): # 每一个点都单独变异
            if np.random.rand() < self.mutate_rate:
                gaussian_noise = np.random.normal(loc=cfg.ADV_PARAMETERS.GN_mean, scale=cfg.ADV_PARAMETERS.GN_std, size=3)
                child[point] += gaussian_noise
        return child

    # def evolve(self, fitness_list):
    #     self.logger.info(" ====> 仅使用fitness差的繁衍后代")
    #     pop_best, pop_leftover = self.select_leftover(fitness_list)
    #     pop_leftover_copy = pop_leftover.copy()
    #     for parent in pop_leftover:  # for every parent
    #         child = self.crossover(parent, pop_leftover_copy)
    #         child = self.mutate(child)
    #         # Epsilon裁剪
    #         vertices = np.array(self.mesh_object.vertices)
    #         delta = np.clip(child - vertices, a_min=-cfg.ADV_PARAMETERS.Epsilon, a_max=cfg.ADV_PARAMETERS.Epsilon)
    #         child = vertices + delta
    #
    #         parent[:] = child
    #     self.pop = np.concatenate((pop_best, pop_leftover), axis=0)

    def evolve(self, fitness_list):
        self.logger.info(" ====> 仅使用fitness好的繁衍后代（自然选择时淘汰fitness差的）")
        pop_best, _ = self.select_leftover(fitness_list)
        pop_best_copy = pop_best.copy()
        for parent in pop_best:  # for every parent
            child = self.crossover(parent, pop_best_copy)
            child = self.mutate(child)
            # Epsilon裁剪
            vertices = np.array(self.mesh_object.vertices)
            delta = np.clip(child - vertices, a_min=-cfg.ADV_PARAMETERS.Epsilon, a_max=cfg.ADV_PARAMETERS.Epsilon)
            child = vertices + delta

            parent[:] = child
        self.pop = np.concatenate((pop_best, pop_best_copy), axis=0)


if __name__ == '__main__':
    mesh = trimesh.load(os.path.join(cfg.PATH.mesh_path))
    # mesh_path = r'./data/cone_0001.ply'
    # mesh = trimesh.load(mesh_path)
    # 打印网格信息
    v = mesh.vertices
    f = mesh.faces
    logger.info("点的维度: {}".format(v.shape))
    logger.info("面的维度: {}".format(f.shape))
    DNA_SIZE = v.shape[0]
    ga = GA(DNA_size=DNA_SIZE,
            cross_rate=cfg.DA_PARAMETERS.CROSS_RATE,
            mutation_rate=cfg.DA_PARAMETERS.MUTATE_RATE,
            pop_size=cfg.DA_PARAMETERS.POP_SIZE,
            mesh=mesh,
            scalar_rate=cfg.ADV_OBJECT.scalar_rate,
            logger=logger
            )

    for generation in range(1, cfg.DA_PARAMETERS.N_GENERATIONS):
        logger.info(' -------------- iteration: {} -----------------'.format(generation))
        fitness_list = ga.get_fitness(generation)
        # fitness = np.random.rand(POP_SIZE)
        ga.evolve(fitness_list)
        best_idx = np.argmin(fitness_list)
        logger.info('Gen: {} | best fit: {:.2f} | best_idx: {}'.format(generation, fitness_list[best_idx], best_idx))

        # 保存best fitness
        save_dir_best_fitness = os.path.join(save_dir_best_fitness_root, 'iter_{}'.format(generation))
        os.makedirs(save_dir_best_fitness, exist_ok=True)
        ga.save_mesh(ga.pop[best_idx, :, :], mesh.faces, os.path.join(save_dir_best_fitness, 'best_fitness-id_{}.ply'.format(best_idx)))

        logger.info('======> YAML_file_path: {}. EXP_NAME: {}'.format(YAML_file_path, cfg.EXP_NAME))
        logger.info('======> car_pro_mean_list: {}'.format(ga.car_pro_mean_list))
        logger.info('======> best_fitness_car_pro_list: {}'.format(ga.best_fitness_car_pro_list))
