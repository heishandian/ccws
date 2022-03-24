import numpy as np
import pandas as pd
import copy
import math
from sklearn.cluster import KMeans


def reshape_obv_to_state(obv, ins, flag):
    rst = []
    for i in range(len(obv)):
        rst.append(obv[i])
    try:
        rst.append(ins['cpu_util_percent'] / 100.0)
        rst.append(ins['mem_util_percent'] / 100.0)
    except IndexError:
        print('error')

    return np.array(rst, dtype=np.float32)


def zoom(data, alpha):
    for i in range(len(data)):
        data[i] = data[i] * alpha
    return data


def reward_modified(data, lower, upper):
    l = len(data)
    if l <= lower:
        return zoom(data, 0.3)
    else:
        alpha = max(0.3, (l - lower) / (upper - lower))
        return zoom(data, alpha)


def min_length_of_trajs(trajs):
    min_length = len(trajs[0]["res"])
    for i in range(len(trajs)):
        if len(trajs[i]["res"]) < min_length:
            min_length = len(trajs[i]["res"])
    return min_length


def get_avg_res_util_from_traj(trajs):
    rur_cpu_trajs = []
    rur_mem_trajs = []
    min_length = min_length_of_trajs(trajs)
    for i in range(len(trajs)):
        rur_cpu_traj = []
        rur_mem_traj = []
        traj_obvs = trajs[i]["res"]
        for j in range(min_length):
            avg_mean = np.array(reshape_obv_to_state(traj_obvs[j], [], False)).mean(axis=0)
            rur_cpu_traj.append(avg_mean[0])
            rur_mem_traj.append(avg_mean[1])
        rur_cpu_trajs.append(rur_cpu_traj)
        rur_mem_trajs.append(rur_mem_traj)
    return np.array(rur_cpu_trajs).mean(axis=0), np.array(rur_mem_trajs).mean(axis=0)


def get_avg_max_res_util_from_traj(trajs):
    max_res_util = []
    for i in range(len(trajs)):
        traj_obvs = trajs[i]["res"]
        avg_mean = np.array(reshape_obv_to_state(traj_obvs[len(traj_obvs) - 1], [], False)).mean(axis=0)
        max_res_util.append([avg_mean[0], avg_mean[1]])
    avg_max_res_util = np.array(max_res_util).mean(0)
    return avg_max_res_util[0], avg_max_res_util[1]


def get_resource_util_and_balance_from_obv(obv):
    rur, rid = 0, 0
    for i in range(len(obv)):
        rur += 1 - obv[i]
    rur = rur / len(obv)
    for i in range(len(obv)):
        rid += math.pow((1 - obv[i] - rur), 2)
    rid = math.sqrt(rid / len(obv))
    return rur, rid


def reward_modified3(data, upper):
    l = len(data)
    alpha = l / upper
    return zoom(data, alpha)


def exp_clustering(exps, cluster_num):
    state_list = []
    for exp in exps:
        state_list.append(exp[0])
    clf = KMeans(n_clusters=cluster_num)
    clf.fit(state_list)
    cls_centers = clf.cluster_centers_
    cls_labels = clf.labels_
    cls_reward_dic = {}

    for i in range(cluster_num):
        cls_reward_dic[i] = []

    for i in range(len(exps)):
        cls_label = cls_labels[i]
        cls_reward_dic[cls_label].append(exps[i][2])

    for cls_reward_index in cls_reward_dic:
        cls_reward_dic[cls_reward_index] = np.mean(cls_reward_dic[cls_reward_index])

    return cls_centers, cls_reward_dic


def get_task_features(cls_centers, cls_reward_dic, task, scheduled_task_num, obv):
    sim_cls, sim_cls_corr = 0, 0
    ids, f1, f2, f3 = task["instance_id"], 0, 0, 777260 - task["time_stamp"] + scheduled_task_num
    cpu_util_sum, mem_util_sum = 0, 0
    state = copy.deepcopy(obv)
    for i in range(len(obv)):
        if i % 2 == 0:
            cpu_util_sum += 1 - obv[i]
        else:
            mem_util_sum += 1 - obv[i]
    task_cpu_util, task_mem_util = task['cpu_util_percent'] / 100.0, task['mem_util_percent'] / 100.0
    f1 = np.corrcoef([task_cpu_util, task_mem_util],
                     [cpu_util_sum / len(obv), mem_util_sum / len(obv)])[0][1]
    state.append(task_cpu_util)
    state.append(task_mem_util)
    for cls_index in range(len(cls_centers)):
        corr = np.corrcoef(state, cls_centers[cls_index])[0][1]
        if corr > sim_cls_corr:
            sim_cls_corr = corr
            sim_cls = cls_index

    f2 = cls_reward_dic[sim_cls]

    return ids, f1, f2, f3


def get_priority_queue(buffer_queue, task_list, scheduled_task_num, obv, pa):
    cls_centers, cls_reward_dic = exp_clustering(buffer_queue, pa.cluster_num)
    ids, f1s, f2s, f3s = [], [], [], []
    for task in task_list:
        id, f1, f2, f3 = get_task_features(cls_centers, cls_reward_dic, task, scheduled_task_num, obv)
        ids.append(id)
        f1s.append(f1)
        f2s.append(f2)
        f3s.append(f3)
    data = pd.DataFrame(
        {'F1': f1s, 'F2': f2s, 'F3': f3s},
        index=ids)
    priority, z, weight = topsis(data, pa.w)
    for task in task_list:
        task['priority'] = priority.S[task['instance_id']]
    task_list.sort(key=lambda x: x["priority"], reverse=True)
    return task_list


def topsis(data, weight=[0, 0.5, 0.5]):
    # 归一化
    data = data / np.sqrt((data ** 2).sum())

    # 最优最劣方案
    Z = pd.DataFrame([data.min(), data.max()], index=['N', 'P'])

    # 距离
    weight = entropyWeight(data) if weight is None else np.array(weight)
    Result = data.copy()
    Result['P'] = np.sqrt(((data - Z.loc['P']) ** 2 * weight).sum(axis=1))
    Result['N'] = np.sqrt(((data - Z.loc['N']) ** 2 * weight).sum(axis=1))

    flag = True
    for dat in Result['N']:
        if dat != 0:
            flag = False
    if flag:
        for i in range(len(Result['N'])):
            Result['N'][i] = 1 / len(Result['N'])

    # 综合得分指数
    Result['P'] = Result['N'] / (Result['N'] + Result['P'])
    Result['S'] = Result.rank(ascending=False)['P']

    return Result, Z, weight


def entropyWeight(data):
    data = np.array(data)
    # 归一化
    P = data / data.sum(axis=0)

    # 计算熵值
    E = np.nansum(-P * np.log(P) / np.log(len(data)), axis=0)

    # 计算权系数
    return (1 - E) / (1 - E).sum()
