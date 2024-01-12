'''
数据生成 DCR
@lizw
'''

import numpy as np
import pandas as pd


#  实验设置
T = 100  # 仿真时长
lamd_list = np.linspace(0.1, 1.0, 10).tolist()  # 业务到达率/子帧
dist = np.random.randint(100, 500, T)  # 船站分布
service_type = 6  # 业务类型2种 实时业务与非实时业务
p = 0.2  # 切换呼叫占总呼叫的比例
# 假设业务服从帕累托分布
packet_min = np.array([15, 30, 45, 15, 30, 45])  # 最小数据包
packet_max = 270  # 最大允许上传数据包大小


def norm(data):
    data_out = data / packet_max
    return data_out


# 生成数据
for i in range(len(lamd_list)):
    lamd_temp = lamd_list[i]
    req_num = np.round(lamd_temp * dist).astype(np.int32)  # 直接采用泊松分布均值 计算请求数
    dataset = []
    for t in range (req_num.shape[0]):  # 遍历每一个子帧
        for j in range(req_num[t]):  # 遍历每一个请求
            subframe = t
            index = j
            st = np.random.choice(service_type)  # 随机选择业务类型
            pt = np.random.rand()
            h = 1 if pt < p else 0  # 是否为切换业务
            # 基于帕累托分布生成数据包大小
            packet = np.round((np.random.pareto(2.5, 1) + 1) * packet_min[st]).astype(np.int32).clip(0, packet_max).item()
            req_temp = {'Subframe': subframe, 'Index': index, 'Service_type': st, 'Handover': h, 'Packet_size': packet}
            dataset.append(req_temp)  # 添加到数据集中
    dataset_pd = pd.DataFrame(dataset)
    dataset_pd['Packet_size_norm'] = norm(dataset_pd['Packet_size'])
    file_path = f'data_{i}.csv'
    dataset_pd.to_csv(file_path, encoding='gbk', index=False)

dist_diff = np.append(dist[1:], dist[0])  # 循环
g_delta = (dist_diff - dist) / dist  # 密度差分变量
g_delta = g_delta[:, np.newaxis]  # 扩充维度
np.save('g_delta.npy', g_delta)

if __name__ == "__main__":
    print('data generate finished')