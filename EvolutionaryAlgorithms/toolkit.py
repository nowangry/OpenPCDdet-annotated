import numpy as np
import torch
import trimesh
import os


def loadPCL(PCL, flag=True):
    if flag:
        PCL = np.fromfile(PCL, dtype=np.float32)
        PCL = PCL.reshape((-1, 4))
    else:
        PCL = pypcd.PointCloud.from_path(PCL)
        PCL = np.array(tuple(PCL.pc_data.tolist()))
        PCL = np.delete(PCL, -1, axis=1)
    return PCL

def save_mesh(vertices, faces, save_path):
    '''
    输入：
        vertices：点云
        faces：构成面的点的索引信息
        save_path：保存路径
    输出：
        mesh：生成的mesh数据
    '''
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    (filedir, filename) = os.path.split(save_path)
    os.makedirs(filedir, exist_ok=True)
    # save
    result = trimesh.exchange.ply.export_ply(mesh, encoding='ascii')
    output_file = open(save_path, "wb+")
    output_file.write(result)
    output_file.close()
    return mesh

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

def record_pred_dicts(logger, writer, pred_dicts_list, generation, scroe_thresh, object_name=''):
    CLASS_NAMES = ['Nothing', 'Car', 'Pedestrian', 'Cyclist']
    total_n = len(pred_dicts_list)
    sum_class = np.zeros(shape=(len(CLASS_NAMES)), dtype=np.int)
    score_lame = 0
    for i in range(total_n):
        if pred_dicts_list[i][0]['pred_labels'].shape[0] > 0:
            score = pred_dicts_list[i][0]['pred_scores'].cpu().numpy()
            label = pred_dicts_list[i][0]['pred_labels'].cpu().numpy()
            sum_class[label] += 1
            # if score < scroe_thresh:
            #     score_lame += 1
        else:
            sum_class[0] += 1

    logger.info('=====> object_name: {}'.format(object_name))
    for i in range(len(CLASS_NAMES)):
        rate = float(sum_class[i]) / total_n
        logger.info('rate: {}/total_n = {}/{} = {:.2f}'.format(CLASS_NAMES[i], sum_class[i], total_n, rate))
        writer.add_scalar('predictions/obj_{}/cls_{}'.format(object_name, CLASS_NAMES[i]), rate, generation)
    # logger.info('score_lame: {}'.format(score_lame))
    # writer.add_scalar('predictions/obj:{}/score_lame'.format(object_name), score_lame, generation)
