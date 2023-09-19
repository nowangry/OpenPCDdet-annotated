import numpy as np
import os
def get_stat_result_from_txt(dir_root, floder_format, fixedEPS_list, attach_rate_list):
    modification_rate = np.zeros(shape=(len(attach_rate_list), len(fixedEPS_list)), dtype=np.float32)
    mAPs = np.zeros_like(modification_rate)
    P_L1 = np.zeros_like(modification_rate)
    P_L2 = np.zeros_like(modification_rate)
    P_Linf = np.zeros_like(modification_rate)
    P_Chamfer = np.zeros_like(modification_rate)
    for i, fixedEPS in enumerate(fixedEPS_list):
        for j, attach_rate in enumerate(attach_rate_list):
            floder = floder_format.format(
                fixedEPS, attach_rate)
            work_dir = os.path.join(dir_root, floder)
            eval_txt = os.path.join(work_dir, 'evaluation.txt')
            if not os.path.exists(eval_txt):
                continue
            with open(eval_txt, 'r') as f:
                data = f.readlines()

                for line in data:
                    if 'mAP: ' in line:
                        mAP = float(line.split('mAP: ')[-1])
                        mAPs[j, i] = mAP
                    elif '修改比例：' in line:
                        mr = float(line.split('修改比例：')[-1].split('%')[0])
                        modification_rate[j, i] = mr
                    elif 'dist_l1: mean=' in line:
                        dist_l1 = float(line.split('dist_l1: mean=')[-1].split(', ')[0])
                        P_L1[j, i] = dist_l1
                    elif 'dist_l2: mean=' in line:
                        dist_l2 = float(line.split('dist_l2: mean=')[-1].split(', ')[0])
                        P_L2[j, i] = dist_l2
                    elif 'dist_l_inf: mean=' in line:
                        dist_l_inf = float(line.split('dist_l_inf: mean=')[-1].split(', ')[0])
                        P_Linf[j, i] = dist_l_inf
                    elif 'dist_chamfer_list: mean=' in line:
                        dist_chamfer = float(line.split('dist_chamfer_list: mean=')[-1].split(', ')[0])
                        P_Chamfer[j, i] = dist_chamfer
                    elif 'dist_hausdorff_list: mean=' in line:
                        break

    return mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer


def get_stat_result_from_txt_KITTI(dir_root, floder_format, fixedEPS_list, attach_rate_list, idx_offset=4):
    modification_rate = np.zeros(shape=(len(attach_rate_list), len(fixedEPS_list)), dtype=np.float32)
    mAPs_Cars = np.zeros_like(modification_rate)
    mAPs_Pedestrian = np.zeros_like(modification_rate)
    mAPs_Cyclist = np.zeros_like(modification_rate)
    P_L1 = np.zeros_like(modification_rate)
    P_L2 = np.zeros_like(modification_rate)
    P_Linf = np.zeros_like(modification_rate)
    P_Chamfer = np.zeros_like(modification_rate)
    for i, fixedEPS in enumerate(fixedEPS_list):
        for j, attach_rate in enumerate(attach_rate_list):
            floder = floder_format.format(
                fixedEPS, attach_rate)
            work_dir = os.path.join(dir_root, floder)

            txt_list = []
            for root, dirs, files in os.walk(work_dir):
                for file in files:
                    if 'evaluation_results' in file:
                        txt_list.append(file)
            txt_list.sort()
            eval_txt = os.path.join(work_dir, txt_list[-1])

            if not os.path.exists(eval_txt):
                continue
            with open(eval_txt, 'r') as f:
                data = f.readlines()
                for idx, line in enumerate(data):
                    if 'Car AP_R40@0.70, 0.70, 0.70:' in line:
                        AP_line = data[idx + idx_offset]
                        if idx_offset == 4:
                            assert '3d   AP:' in AP_line
                        elif idx_offset == 3:
                            assert 'bev  AP:' in AP_line
                        AP = np.array(AP_line.split(':')[-1].split(',')[:]).astype(np.float)
                        mAPs_Cars[j, i] = AP.mean()
                    elif 'Pedestrian AP_R40@0.50, 0.50, 0.50:' in line:
                        AP_line = data[idx + idx_offset]
                        if idx_offset == 4:
                            assert '3d   AP:' in AP_line
                        elif idx_offset == 3:
                            assert 'bev  AP:' in AP_line
                        AP = np.array(AP_line.split(':')[-1].split(',')[:]).astype(np.float)
                        mAPs_Pedestrian[j, i] = AP.mean()
                    if 'Cyclist AP_R40@0.50, 0.50, 0.50:' in line:
                        AP_line = data[idx + idx_offset]
                        if idx_offset == 4:
                            assert '3d   AP:' in AP_line
                        elif idx_offset == 3:
                            assert 'bev  AP:' in AP_line
                        AP = np.array(AP_line.split(':')[-1].split(',')[:]).astype(np.float)
                        mAPs_Cyclist[j, i] = AP.mean()
                    elif '修改比例：' in line:
                        mr = float(line.split('修改比例：')[-1].split('%')[0])
                        modification_rate[j, i] = mr
                    elif 'dist_l1: mean=' in line:
                        dist_l1 = float(line.split('dist_l1: mean=')[-1].split(', ')[0])
                        P_L1[j, i] = dist_l1
                    elif 'dist_l2: mean=' in line:
                        dist_l2 = float(line.split('dist_l2: mean=')[-1].split(', ')[0])
                        P_L2[j, i] = dist_l2
                    elif 'dist_l_inf: mean=' in line:
                        dist_l_inf = float(line.split('dist_l_inf: mean=')[-1].split(', ')[0])
                        P_Linf[j, i] = dist_l_inf
                    elif 'dist_chamfer_list: mean=' in line:
                        dist_chamfer = float(line.split('dist_chamfer_list: mean=')[-1].split(', ')[0])
                        P_Chamfer[j, i] = dist_chamfer
                    elif 'dist_hausdorff_list: mean=' in line:
                        break

    mAPs = (mAPs_Cars + mAPs_Pedestrian + mAPs_Cyclist) / 3.0
    return mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer

def get_stat_result_from_txt_KITTI_bev(dir_root, floder_format, fixedEPS_list, attach_rate_list):
    modification_rate = np.zeros(shape=(len(attach_rate_list), len(fixedEPS_list)), dtype=np.float32)
    mAPs_Cars = np.zeros_like(modification_rate)
    mAPs_Pedestrian = np.zeros_like(modification_rate)
    mAPs_Cyclist = np.zeros_like(modification_rate)
    P_L1 = np.zeros_like(modification_rate)
    P_L2 = np.zeros_like(modification_rate)
    P_Linf = np.zeros_like(modification_rate)
    P_Chamfer = np.zeros_like(modification_rate)
    for i, fixedEPS in enumerate(fixedEPS_list):
        for j, attach_rate in enumerate(attach_rate_list):
            floder = floder_format.format(
                fixedEPS, attach_rate)
            work_dir = os.path.join(dir_root, floder)
            idx_offset = 3
            txt_list = []
            for root, dirs, files in os.walk(work_dir):
                for file in files:
                    if 'evaluation_results' in file:
                        txt_list.append(file)
            txt_list.sort()
            eval_txt = os.path.join(work_dir, txt_list[-1])

            if not os.path.exists(eval_txt):
                continue
            with open(eval_txt, 'r') as f:
                data = f.readlines()
                for idx, line in enumerate(data):
                    if 'Car AP_R40@0.70, 0.70, 0.70:' in line:
                        AP_line = data[idx + idx_offset]
                        if idx_offset == 4:
                            assert '3d   AP:' in AP_line
                        elif idx_offset == 3:
                            assert 'bev  AP:' in AP_line
                        AP = np.array(AP_line.split(':')[-1].split(',')[:]).astype(np.float)
                        mAPs_Cars[j, i] = AP[1]
                    elif 'Pedestrian AP_R40@0.50, 0.50, 0.50:' in line:
                        AP_line = data[idx + idx_offset]
                        if idx_offset == 4:
                            assert '3d   AP:' in AP_line
                        elif idx_offset == 3:
                            assert 'bev  AP:' in AP_line
                        AP = np.array(AP_line.split(':')[-1].split(',')[:]).astype(np.float)
                        mAPs_Pedestrian[j, i] = AP[1]
                    if 'Cyclist AP_R40@0.50, 0.50, 0.50:' in line:
                        AP_line = data[idx + idx_offset]
                        if idx_offset == 4:
                            assert '3d   AP:' in AP_line
                        elif idx_offset == 3:
                            assert 'bev  AP:' in AP_line
                        AP = np.array(AP_line.split(':')[-1].split(',')[:]).astype(np.float)
                        mAPs_Cyclist[j, i] = AP[1]
                    elif '修改比例：' in line:
                        mr = float(line.split('修改比例：')[-1].split('%')[0])
                        modification_rate[j, i] = mr
                    elif 'dist_l1: mean=' in line:
                        dist_l1 = float(line.split('dist_l1: mean=')[-1].split(', ')[0])
                        P_L1[j, i] = dist_l1
                    elif 'dist_l2: mean=' in line:
                        dist_l2 = float(line.split('dist_l2: mean=')[-1].split(', ')[0])
                        P_L2[j, i] = dist_l2
                    elif 'dist_l_inf: mean=' in line:
                        dist_l_inf = float(line.split('dist_l_inf: mean=')[-1].split(', ')[0])
                        P_Linf[j, i] = dist_l_inf
                    elif 'dist_chamfer_list: mean=' in line:
                        dist_chamfer = float(line.split('dist_chamfer_list: mean=')[-1].split(', ')[0])
                        P_Chamfer[j, i] = dist_chamfer
                    elif 'dist_hausdorff_list: mean=' in line:
                        break

    # mAPs = (mAPs_Cars + mAPs_Pedestrian + mAPs_Cyclist) / 3.0
    mAPs = mAPs_Cars
    return mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer

def get_stat_result_from_txt_waymo(dir_root, floder_format, fixedEPS_list, attach_rate_list):
    mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer = get_stat_result_from_txt(dir_root, floder_format,
                                                                                      fixedEPS_list, attach_rate_list)

    for i, fixedEPS in enumerate(fixedEPS_list):
        for j, attach_rate in enumerate(attach_rate_list):
            floder = floder_format.format(
                fixedEPS, attach_rate)
            work_dir = os.path.join(dir_root, floder)
            eval_txt = os.path.join(work_dir, 'evaluation_waymo.txt')
            if not os.path.exists(eval_txt):
                continue
            with open(eval_txt, 'r') as f:
                data = f.readlines()

                for line in data:
                    if 'LEVEL_2 AP: Average: ' in line:
                        mAP = float(line.split('LEVEL_2 AP: Average: ')[-1].split('%')[0])
                        mAPs[j, i] = mAP

    return mAPs, modification_rate, P_L1, P_L2, P_Linf, P_Chamfer