import openmesh
import trimesh

def test(mesh_path):
    print("meth_path = {}".format(mesh_path))
    mesh1 = openmesh.read_trimesh(mesh_path)
    # 打印网格信息
    # 获取顶点、边、面的总数
    print('openmesh')
    print('顶点总数：', mesh1.n_vertices())
    print('面总数  ：', mesh1.n_faces())
    print('边总数  ：', mesh1.n_edges())
    print('------------------------------')



    mesh2 = trimesh.load(mesh_path)
    # 打印网格信息
    v = mesh2.vertices
    f = mesh2.faces
    print("trimesh")
    print("点的维度: {}".format(v.shape))
    print("面的维度: {}".format(f.shape))
    print('------------------------------')



mesh_path = r'../data/GeneticAlgorithm/chair-origin.ply'
test(mesh_path)
mesh_path = r'../data/GeneticAlgorithm/cone_0001.ply'
test(mesh_path)