"""
Visualize Genetic Algorithm to find the shortest path for travel sales problem.

Visit my tutorial website for more: https://mofanpy.com/tutorials/
"""
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d
import openmesh as om
import pypcd as pypcd
from openmesh import *
import os
from lidar_simulation.lidar import Lidar
import demo_GA_Attack
import torch

DNA_size = None  # DNA size
CROSS_RATE = 0.1
MUTATE_RATE = 0.2
MUTATE_STD = 0.05
POP_SIZE = 160
N_GENERATIONS = 1000
data_root = r'/data/dataset_wujunqi/GeneticAlgorithm'


def loadPCL(self, PCL, flag=True):
    if flag:
        PCL = np.fromfile(PCL, dtype=np.float32)
        PCL = PCL.reshape((-1, 4))
    else:
        PCL = pypcd.PointCloud.from_path(PCL)
        PCL = np.array(tuple(PCL.pc_data.tolist()))
        PCL = np.delete(PCL, -1, axis=1)
    return PCL

# f-function in the paper
def CW_f(self, outputs, labels):
    # one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)
    one_hot_labels = torch.eye(len(outputs[0]))[labels]

    i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)  # get the second largest logit
    j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit

    if self._targeted:
        return torch.clamp((i - j), min=-self.kappa)
    else:
        return torch.clamp((j - i), min=-self.kappa)


def L2_mesh(mesh1, mesh2):
    point_array1 = mesh1.points()
    point_array2 = mesh2.points()


class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, mesh):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.mesh_origin = mesh
        self.pop = np.zeros(shape=(1, DNA_size, 3), dtype=np.float64)
        cnt = 0
        for vertex in mesh.vertices():
            [x, y, z] = mesh.point(vertex)
            self.pop[0, cnt, :] = [x, y, z]
            cnt += 1
        pop_test = mesh.points()
        assert (pop_test == self.pop).all()
        self.pop = np.repeat(self.pop, repeats=pop_size, axis=0) # 复制
        # 添加高斯噪声
        gaussian_noise = np.random.normal(loc=0.0, scale=0.1, size=self.pop.shape)
        self.pop = self.pop + gaussian_noise

    def translate_mesh(self, mesh_list):
        for p in range(self.pop_size):
            mesh = TriMesh()
            for point in range(self.DNA_size):
                [x, y, z] = self.pop[p, point, :]
                mesh.add_vertex([x, y, z])
            for face in self.mesh_origin.face():
                mesh.add_face(face)
            mesh_list.appand(mesh)

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
        PC_path = r'../data/lidar-background.bin'
        point_cloud_background = loadPCL(PCL_path, True)
        # 3
        # 生成点云并保存
        for p in range(self.pop_size):
            vertices = mesh_list[p].vertices()
            polygons = mesh_list[p].faces()
            point_cloud_object = Lidar(delta_azimuth=2 * np.pi / 2000,
                                delta_elevation=np.pi / 800,
                                position=(0, -10, 0)).sample_3d_model_gpu(vertices, polygons)
            point_cloud_input = np.concatenate((point_cloud_object, point_cloud_background), axis=0)
            bin_file = os.path.join(data_root, generation, str(p) + '.bin')
            point_cloud_input.tofile(bin_file)
        # 点云输入网络
        pred_dicts_list, recall_dicts_list, batch_dict_list = demo_GA_Attack.solve()

        target_label = 0
        fitness_list = []
        for p in range(self.pop_size):
            rcnn_cls = batch_dict_list[p]['rcnn_cls']
            f_cw = CW_f(rcnn_cls, target_label)
            loss_cw = f_cw + L2_mesh()




        return fitness




    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)                        # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool) # choose crossover points
            parent[cross_points] = pop[i_, cross_points]
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size): # 每一个点都单独变异
            if np.random.rand() < self.mutate_rate:
                gaussian_noise = np.random.normal(loc=0.0, scale=0.1, size=3)
                child[point] += gaussian_noise
        return child

    def evolve(self, fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop


if __name__ == '__main__':

    demo_GA_Attack.setup(data_root)

    # mesh_path = r'../data/GeneticAlgorithm/chair-origin.ply'
    # mesh = o3d.io.read_triangle_mesh(mesh_path)
    # print(mesh)
    # # 打印网格定点信息
    # print('Vertices:')
    # print(np.asarray(mesh.vertices))
    # # 打印网格的三角形
    # print('Triangles:')
    # print(np.asarray(mesh.triangles))

    mesh_path = r'../data/GeneticAlgorithm/chair-origin.ply'
    mesh = om.read_trimesh(mesh_path)
    # 打印网格信息
    # 获取顶点、边、面的总数
    print('顶点总数：', mesh.n_vertices())
    print('面总数  ：', mesh.n_faces())
    print('边总数  ：', mesh.n_edges())
    # 遍历所有的顶点，获取每个vertex的坐标
    for vertex in mesh.vertices():
        print('顶点的数据类型：', type(vertex), '顶点坐标：', mesh.point(vertex), '顶点坐标的数据类型', type(mesh.point(vertex)))
        break
    # 遍历所有的边和面
    for edge in mesh.edges():
        print(type(edge))
        break
    for face in mesh.faces():
        print(type(face))
        break


    DNA_size = mesh.n_vertices()
    ga = GA(DNA_size=DNA_size, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE, mesh=mesh)

    for generation in range(N_GENERATIONS):
        # lx, ly = ga.translateDNA(ga.pop, env.city_position)
        # fitness, total_distance = ga.get_fitness(lx, ly)
        fitness = ga.get_fitness(generation)
        # fitness = np.random.rand(POP_SIZE)
        ga.evolve(fitness)
        best_idx = np.argmax(fitness)
        print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)
