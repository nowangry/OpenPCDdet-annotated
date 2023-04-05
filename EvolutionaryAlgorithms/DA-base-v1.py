import numpy as np
# import open3d as o3d
# import openmesh as om
# from openmesh import *
import os
from lidar_simulation.lidar import Lidar
import demo_EA_Attack
import torch
import trimesh
import sys
import datetime
from torch.utils.tensorboard import SummaryWriter
import random
from toolkit import *
import matplotlib.pyplot as plt

# 实验配置文件 ----------------------------------------------
# YAML_file_path = r'./cfgs/DA-exp_2.3.1-object.yaml'
YAML_file_path = r'./cfgs/DA-debug.yaml'


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

# os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.CUDA_VISIBLE_DEVICES)
# --------------------------------------------------------

class EA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size,
                 leftover_rate=0.5, logger=None):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.leftover_rate = leftover_rate
        self.logger = logger
        self.writer = SummaryWriter(tensorboard_path)
        self.car_pro_mean_list = []
        self.best_fitness_car_pro_list = []
        self.point_cloud_background = loadPCL(PC_background_path, True)  # mean:0.2641031, std:0.12991029
        self.intensity_mean = np.mean(self.point_cloud_background[:, 3])

    def init_population(self, mesh):
        # 预处理 -----------------
        vertices = np.array(mesh.vertices)
        print('原始位置：')
        show_width_height(vertices)

        vertices *= cfg.ADV_OBJECT.scalar_rate
        print("缩放后：")
        show_width_height(vertices)

        for i in range(len(off_set_list)):
            vertices[:, i] += off_set_list[i]
        print("加偏移量 {} 后：".format(off_set_list))
        show_width_height(vertices)
        # 预处理 -----------------
        self.object_location_path = os.path.join(r'./data/outputs', cfg.EXP_NAME, 'object_location.ply')
        self.mesh_object = save_mesh(vertices, mesh.faces, self.object_location_path)

        vertices = np.expand_dims(vertices, axis=0)
        assert len(vertices.shape) == 3
        population = np.repeat(vertices, repeats=self.pop_size, axis=0) # 复制
        # 添加高斯噪声
        gaussian_noise = np.random.normal(loc=cfg.ADV_PARAMETERS.GN_mean, scale=cfg.ADV_PARAMETERS.GN_std, size=population.shape)
        pop_adv = population + gaussian_noise
        delta = np.clip(pop_adv - population, a_min=-cfg.ADV_PARAMETERS.Epsilon, a_max=cfg.ADV_PARAMETERS.Epsilon)
        population = population + delta
        return population

    def get_target_object_features(self):
        self.logger.info(" 对抗物体：mesh --> 点云")
        bin_file_object = self.rendering_and_save(self.object_location_path, False)
        pred_dicts_list, batch_dict_list, _ = demo_EA_Attack.network_forward(self.logger, cfg, bin_file_object)
        self.logger.info("pred_dicts_list: {}".format(pred_dicts_list))
        # 测试一下原始object
        save_dir = os.path.join(save_dir_root, 'validation', 'iter_{}'.format(0))
        self.evaluation(self.object_location_path, save_dir, cfg.VAL.sample_num, 0, cfg.ADV_OBJECT_NAME)


        self.logger.info(" 目标类别：mesh --> 点云")
        mesh_target = trimesh.load(target_mesh_path)
        vertices = np.array(mesh_target.vertices)
        vertices *= cfg.TARGET_OBJECT.scalar_rate
        for i in range(len(off_set_list)):
            vertices[:, i] += off_set_list[i]
        print("加偏移量 {} 后：".format(off_set_list))
        show_width_height(vertices)
        self.target_location_path = os.path.join(r'./data/outputs', cfg.EXP_NAME, 'target_location.ply')
        self.mesh_target = save_mesh(vertices, mesh_target.faces, self.target_location_path)

        bin_file_target = self.rendering_and_save(self.target_location_path, False)
        pred_dicts_list, batch_dict_list, _ = demo_EA_Attack.network_forward(self.logger, cfg, bin_file_target)
        self.logger.info("pred_dicts_list: {}".format(pred_dicts_list))
        # 测试一下原始target
        save_dir = os.path.join(save_dir_root, 'validation', 'iter_{}-target'.format(0))
        self.evaluation(self.target_location_path, save_dir, cfg.VAL.sample_num, 0, cfg.TARGET_NAME)

        self.target_features = batch_dict_list[0]['point_features']
        self.target_rpn_roi = batch_dict_list[0]['rois']
        self.target_rcnn_roi = batch_dict_list[0]['batch_box_preds']

    def rendering_mesh(self, mesh_path=None, is_off_set=False, mesh=None, lidar_location=[0, 0, 0]):
        '''
        输入：
            mesh_path：mesh路径
            is_off_set：是否需要加上偏置
            mesh: 现成的mesh数据
        输出：
            对mesh经过lidar渲染之后的点云
        '''
        if (mesh_path is not None) and (mesh is None):
            mesh = trimesh.load(mesh_path)
        # 打印网格信息
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        if (mesh_path is not None) and (mesh is None):
            self.logger.info("---- 点的维度: {}".format(vertices.shape))
            self.logger.info("---- 面的维度: {}".format(faces.shape))
        if is_off_set:
            # 加偏置
            for i in range(len(off_set_list)):
                vertices[:, i] += off_set_list[i]

        # lidar渲染 默认方位角精读：0.18，仰角精度：0.225
        point_cloud_object_render = Lidar(delta_azimuth=cfg.LIDAR.delta_azimuth / 360.0 * 2 * np.pi,
                                          delta_elevation=cfg.LIDAR.delta_elevation / 360.0 * 2 * np.pi,
                                          position=lidar_location).sample_3d_model_gpu(vertices, faces)  # 确认渲染器使用
        intensity_array = np.ones((point_cloud_object_render.shape[0], 1), dtype=np.float32) * self.intensity_mean
        point_cloud_input = np.concatenate((point_cloud_object_render, intensity_array), axis=1)
        return point_cloud_input

    def rendering_and_save(self, mesh_path, is_off_set):
        '''
        输入：
            mesh_path：mesh路径
            is_off_set：是否需要加上偏置
        输出：
            保存mesh经过lidar渲染之后的点云
        '''
        point_cloud_input = self.rendering_mesh(mesh_path=mesh_path, is_off_set=is_off_set, lidar_location=lidar_position)
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

    def translate_mesh(self, mesh_list, population):
        faces = np.array(self.mesh_object.faces)
        vertices = np.zeros((self.DNA_size, 3), dtype=np.float32)
        for p in range(self.pop_size):
            for point in range(self.DNA_size):
                [x, y, z] = population[p, point, :]
                vertices[point, :] = [x, y, z]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh_list.append(mesh)

    def calculate_fitness(self, population, generation, lidar_location):
        '''
        1.使用点和面的关系构建对抗物体mesh
        2.对抗物体mesh输入到Lidar渲染器得到对抗物体点云
        3.（对抗物体点云 + 背景点云)输入PointRCNN得到预测结果
        4.根据预测结果计算fitness
        '''
        # 1
        mesh_list = []
        self.translate_mesh(mesh_list, population)
        # 2
        # point_cloud_background = loadPCL(PC_background_path, True) # mean:0.2641031, std:0.12991029
        # intensity_mean = np.mean(point_cloud_background[:, 3])
        # 3
        # 生成点云并保存
        save_dir = os.path.join(save_dir_root, 'Generate', 'iter' + str(generation))
        os.makedirs(save_dir, exist_ok=True)
        for p in range(self.pop_size):
            vertices = mesh_list[p].vertices
            polygons = mesh_list[p].faces
            point_cloud_object_render = Lidar(delta_azimuth=cfg.LIDAR.delta_azimuth / 360.0 * 2 * np.pi,
                                delta_elevation=cfg.LIDAR.delta_elevation / 360.0 * 2 * np.pi,
                                position=lidar_location).sample_3d_model_gpu(vertices, polygons) # 确认渲染器使用
            intensity_array = np.ones((point_cloud_object_render.shape[0], 1), dtype=np.float32) * self.intensity_mean
            point_cloud_object = np.concatenate((point_cloud_object_render, intensity_array), axis=1)
            point_cloud_input = point_cloud_object
            bin_file = os.path.join(save_dir, 'iter_' + str(generation) + '-popID_' + str(p) + '.bin')
            point_cloud_input = point_cloud_input.astype(np.float32)
            point_cloud_input.tofile(bin_file)

        self.logger.info("====> 对抗物体维度 point_cloud_input.shape: {}".format(point_cloud_input.shape))

        # 点云输入网络
        pred_dicts_list, batch_dict_list, car_probability_list = demo_EA_Attack.network_forward(self.logger, cfg, save_dir)

        record_pred_dicts(self.logger, self.writer, pred_dicts_list, generation, cfg.MODEL.POST_PROCESSING.SCORE_THRESH, cfg.ADV_OBJECT_NAME)

        car_pro_mean = np.array(car_probability_list).mean()
        car_pro_std = np.array(car_probability_list).std()
        self.car_pro_mean_list.append(car_pro_mean)
        self.logger.info('iter: {}. 分类为车的平均概率为： {:.3f}%. STD为：{:.3f}'.format(generation, car_pro_mean * 100, car_pro_std))
        self.writer.add_scalar('car_probability/car_p_mean', car_pro_mean, generation)
        self.writer.add_scalar('car_probability/car_p_std', car_pro_mean, generation)

        fitness_list = []
        loss_cls_list = []
        loss_features_list = []
        loss_box_list = []
        loss_L2_list = []
        Z_bg_list = []
        Z_rpn_list = []
        Z_t_list = []
        for p in range(self.pop_size):
            # Loss_cls
            Z_bg = batch_dict_list[p]['batch_cls_preds'][0, :, 0] # [1, 100, 1]. rcnn背景分数 ?
            Z_rpn = batch_dict_list[p]['roi_scores'][0] # [1, 100]. rpn oojectiveness scores
            Z_t = batch_dict_list[p]['roi_cls_logit'][0, :, cfg.ADV_PARAMETERS.Target_Label] # [1, 100, 3]   分类成目标类别的logits
            k_threshold = 0.9
            loss_cls = ((Z_t[Z_t < k_threshold]) * (Z_bg - Z_rpn - Z_t)).sum()
            # debug
            Z_bg_list.append(((Z_t[Z_t < k_threshold]) * Z_bg).sum().cpu())
            Z_rpn_list.append(-((Z_t[Z_t < k_threshold]) * Z_rpn).sum().cpu())
            Z_t_list.append(-((Z_t[Z_t < k_threshold]) * Z_t).sum().cpu())

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

        # loss_cls = ((Z_t[Z_t < k_threshold]) * (Z_bg - Z_rpn - Z_t)).sum()
        self.logger.info('loss/loss_cls/Z_bg: mean={:.3f}, std={:.3f}'.format(np.array(Z_bg_list).mean(), np.array(Z_bg_list).std()))
        self.logger.info('loss/loss_cls/Z_rpn: mean={:.3f}, std={:.3f}'.format(np.array(Z_rpn_list).mean(), np.array(Z_rpn_list).std()))
        self.logger.info('loss/loss_cls/Z_t: mean={:.3f}, std={:.3f}'.format(np.array(Z_t_list).mean(), np.array(Z_t_list).std()))
        self.writer.add_scalar('loss/loss_cls/Z_bg', np.array(Z_bg_list).mean(), generation)
        self.writer.add_scalar('loss/loss_cls/Z_rpn', np.array(Z_rpn_list).mean(), generation)
        self.writer.add_scalar('loss/loss_cls/Z_t', np.array(Z_t_list).mean(), generation)


        best_idx = np.argmin(fitness_array)
        best_fitness_car_pro = car_probability_list[best_idx]
        self.best_fitness_car_pro_list.append(best_fitness_car_pro)
        self.writer.add_scalar('car_probability/best_fitness', best_fitness_car_pro, generation)
        return fitness_array

    def crossover_binary(self, Mpopulation, population):
        Cpopulation = np.zeros((self.pop_size, self.DNA_size, 3))
        for i in range(self.pop_size):
            rand_j = random.randint(0, self.DNA_size - 1)
            for j in range(self.DNA_size):
                rand_float = random.random()
                if rand_float <= self.cross_rate or rand_j == j:
                    Cpopulation[i, j] = Mpopulation[i, j]
                else:
                    Cpopulation[i, j] = population[i, j]
        return Cpopulation

    def mutation_rand(self, population):
        Mpopulation = np.zeros((self.pop_size, self.DNA_size, 3))
        for i in range(self.pop_size):
            r1 = r2 = r3 = 0
            while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
                r1 = random.randint(0, self.pop_size - 1)
                r2 = random.randint(0, self.pop_size - 1)
                r3 = random.randint(0, self.pop_size - 1)
            Mpopulation[i] = population[r1] + cfg.DA_PARAMETERS.F_mutation * (population[r2] - population[r3])

            # Epsilon裁剪
            vertices = np.array(self.mesh_object.vertices)
            delta = np.clip(Mpopulation[i] - vertices, a_min=-cfg.ADV_PARAMETERS.Epsilon, a_max=cfg.ADV_PARAMETERS.Epsilon)
            Mpopulation[i] = vertices + delta

        return Mpopulation

    def selection(self, Cpopulation, population, pfitness, generation):
        Cfitness = self.calculate_fitness(Cpopulation, generation, lidar_position)
        for i in range(self.pop_size):
            if Cfitness[i] < pfitness[i]:
                population[i] = Cpopulation[i]
                pfitness[i] = Cfitness[i]
            else:
                population[i] = population[i]
                pfitness[i] = pfitness[i]
        return population, pfitness

    def main_process(self, mesh):
        optimization = []
        # eval_path_list = []

        population = self.init_population(mesh)  # 种群初始化
        self.get_target_object_features() # 得到被攻击的目标类别的特征

        fitness = self.calculate_fitness(population, 0, lidar_position)
        optimization.append(min(fitness))
        Best_indi_index = np.argmin(fitness)
        Best_indi = population[Best_indi_index, :, :]
        for step in range(1, cfg.DA_PARAMETERS.N_GENERATIONS + 1):
            Mpopulation = self.mutation_rand(population)  # 变异
            Cpopulation = self.crossover_binary(Mpopulation, population)
            population, fitness = self.selection(Cpopulation, population, fitness, step)
            optimization.append(min(fitness))
            Best_indi_index = np.argmin(fitness)
            Best_indi_pc = population[Best_indi_index, :, :]

            # 保存信息
            self.logger.info('Gen: {} | best fit: {:.2f} | best_idx: {}'.format(step, fitness[Best_indi_index], Best_indi_index))
            self.logger.info('======> YAML_file_path: {}. EXP_NAME: {}'.format(YAML_file_path, cfg.EXP_NAME))
            self.logger.info('======> car_pro_mean_list: {}'.format(self.car_pro_mean_list))
            self.logger.info('======> best_fitness_car_pro_list: {}'.format(self.best_fitness_car_pro_list))
            # 保存best fitness
            save_dir_best_fitness = os.path.join(save_dir_best_fitness_root, 'iter_{}'.format(step))
            os.makedirs(save_dir_best_fitness, exist_ok=True)
            Best_mesh_path = os.path.join(save_dir_best_fitness, 'best_fitness-id_{}.ply'.format(Best_indi_index))
            save_mesh(Best_indi_pc, self.mesh_object.faces, Best_mesh_path)

            if step % cfg.VAL.val_frequency == 0:
                self.logger.info("------------------------ Runing validation ----------------------")
                save_dir = os.path.join(save_dir_root, 'validation', 'iter_{}'.format(step))
                self.evaluation(Best_mesh_path, save_dir, cfg.VAL.sample_num, step, cfg.ADV_OBJECT_NAME)

        plt.plot(optimization, '^-', markersize=10)
        plt.title("fitness")
        plt.show()
        print(optimization)
        return Best_mesh_path


    def evaluation(self, mesh_path, save_dir, sample_num, generation, object_name=''):
        lidar_location = cfg.EVAL.lidar_location
        np.random.seed(cfg.EVAL.random_seed)
        object_locations = np.zeros(shape=(sample_num, 3), dtype=np.float32)
        object_locations[:, 0] = np.random.uniform(low=cfg.EVAL.object_location_area.x[0], high=cfg.EVAL.object_location_area.x[1], size=sample_num)
        object_locations[:, 1] = np.random.uniform(low=cfg.EVAL.object_location_area.y[0], high=cfg.EVAL.object_location_area.y[1], size=sample_num)
        object_locations[:, 2] = np.random.uniform(low=cfg.EVAL.object_location_area.z[0], high=cfg.EVAL.object_location_area.z[1], size=sample_num)

        # 生成测试样本
        # 1 调整物体位置到[0, 0, 0]
        mesh = trimesh.load(mesh_path)
        for i in range(len(off_set_list)):
            mesh.vertices[:, i] -= off_set_list[i]

        # 2 生成测试样本
        vertices = np.array(mesh.vertices)
        os.makedirs(save_dir, exist_ok=True)
        for sample_index in range(sample_num):
            mesh.vertices = vertices + object_locations[sample_index, :]
            save_path = os.path.join(save_dir, 'sample_{}-x{:.2f}_y{:.2f}_z{:.2f}.bin'.format(sample_index,
                                                                                              object_locations[sample_index, 0],
                                                                                              object_locations[sample_index, 1],
                                                                                              object_locations[sample_index, 2]))
            point_clouds = self.rendering_mesh(mesh=mesh, lidar_location=lidar_location)
            point_clouds = point_clouds.astype(np.float32)
            point_clouds.tofile(save_path)

        # 3 测试样本输入到网络评测
        pred_dicts_list, batch_dict_list, car_probability_list = demo_EA_Attack.network_forward(self.logger, cfg, save_dir)
        self.logger.info('car_probability_list: {}'.format(car_probability_list))
        self.logger.info('car_probability_list: mean={:.3f}, std={:.3f}'.format(np.array(car_probability_list).mean(), np.array(car_probability_list).std()))
        record_pred_dicts(self.logger, self.writer, pred_dicts_list, generation, cfg.MODEL.POST_PROCESSING.SCORE_THRESH, object_name)

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
    ea = EA(DNA_size=DNA_SIZE,
            cross_rate=cfg.DA_PARAMETERS.CROSS_RATE,
            mutation_rate=cfg.DA_PARAMETERS.MUTATE_RATE,
            pop_size=cfg.DA_PARAMETERS.POP_SIZE,
            logger=logger
            )
    Best_mesh_path = ea.main_process(mesh)
    logger.info("------------------------ Runing evalation ----------------------")
    ea.evaluation(Best_mesh_path, os.path.join(save_dir_root, 'evalation'), cfg.EVAL.sample_num,
                  cfg.DA_PARAMETERS.N_GENERATIONS, cfg.ADV_OBJECT_NAME)

