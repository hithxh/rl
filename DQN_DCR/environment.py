import numpy as np
import pandas as pd
import copy

class Env:
    def __init__(self, env_name, algo, cum_length):
        # 环境设置
        self.name = env_name  # 数据名称
        self.algo = algo
        self.access_mode = "best"  # 随机接入
        self.rao_num = 72  # 随机接入机会数
        self.dc_num = 16  # 一个子帧中DC数量
        self.max_slot = 270  # 最多传输270个时隙
        self.slot_d = 30  # 不同DC时隙数量的间隔
        # 规则设置
        self.observation_space = 3  # 状态空间元素个数
        self.action_dim = 4  # 动作变量维度
        self.service_w = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0]) # 业务优先级收益
        self.handover_c = 0.2  # 新请求相对于切换请求的优先级
        # 读取数据
        data_path = f'./data/{self.name}.csv'  # 数据路径
        self.req_list = pd.read_csv(data_path, sep=',', index_col=False)  # 获取数据
        self.g_delta = np.load(f'./data/g_delta.npy')  # 获取TDD变量
        self.T = self.g_delta.shape[0]  # 获得仿真时长
        # 初始化信道利用矩阵信息
        # 特征： 类型，切换，倒数计时器
        cum1 = np.ones((self.T, self.dc_num, 1)) * 6  # 特征： 类型
        cum2 = np.ones((self.T, self.dc_num, 1)) * 2  # 特征： 切换
        cum3 = np.zeros((self.T, self.dc_num, 1))  # 特征： 倒数计时器
        self.cum = np.concatenate([cum1, cum2, cum3], axis=2)
        self.cum_length = cum_length  # 采用cum_length个历史数据

    def rac(self, n):
        # 模拟随机接入过程
        # 直接采用概率模拟
        if self.access_mode == 'baseline':
            p_s = np.power((1 - 1 / self.rao_num), n - 1)
            ex_success_num = np.round(n * p_s).astype(np.int32)  # 期望的成功接入数量
        elif self.access_mode == 'best':
            n_p = min(n, self.rao_num)  # 当n大于接入机会时，采用最优ACB策略控制
            p_s = np.power((1 - 1 / self.rao_num), n_p - 1)
            ex_success_num = np.round(n_p * p_s).astype(np.int32)  # 期望的成功接入数量
        else:
            raise ValueError('invalid access mode')
        idx_success = np.random.choice(np.arange(n), ex_success_num, replace=False)  # 成功接入的索引值
        return idx_success

    def reset(self, seed=10):

        # cum_w 信道利用矩阵序列
        cum_w = copy.deepcopy(np.append(self.cum[self.T - self.cum_length + 1:self.T], self.cum[0:1], axis=0))
        avail_dc = np.arange(self.dc_num)[self.cum[0, :, 2] == 0]  # 未被占用的DC
        avail1 = (avail_dc[avail_dc < 8] + 1).tolist()
        avail2 = (avail_dc[(avail_dc > 7) & (avail_dc < 12)] + 1).tolist()
        avail3 = (avail_dc[avail_dc > 11] + 1).tolist()
        avail = (avail1, avail2, avail3)
        # 请求信息
        req_list = self.req_list[self.req_list['Subframe'] == 0]  # 选择初始子帧数据
        req_array = np.array(req_list)  # 转换array
        req_info = req_array[:, [2, 3, 5]]  # 选取有效信息作为 m^t_i  [index,st,h,packet,packet_norm]
        # #随机接入过程
        success_idx = self.rac(req_array.shape[0])  # 接入成功索引
        req_success_info = req_info[success_idx]  # 接入请求信息
        assert req_success_info.shape[0] > 0, 'no req'
        # TDD
        # g_delta_w = np.zeros((self.cum_length, 1))  # TDD变量 与信道利用矩阵对应
        # TDD变量 与信道利用矩阵对应
        g_delta_w = np.append(self.g_delta[self.T - self.cum_length + 1:self.T], self.g_delta[0:1], axis=0)
        g_delta_w = np.expand_dims(np.tile(g_delta_w, (1, self.dc_num)), axis=2)  # 复制dc_num次

        state = (cum_w, req_success_info, g_delta_w)

        return state, avail

    def step(self, state, action_, avail, t):
        """
        action: n_t * 4
        avail: 元组，（[1,2,3],[4,5,6],[7,8]）
        step t:t+1
        """
        # 奖励
        action = copy.deepcopy(action_)  # 防止修改 replay中的action  action 为可变变量
        req_success_info = state[1]  # 获取接入请求信息
        d = req_success_info[:, 2] * self.max_slot / ((action + 1e-5) * self.slot_d)  # 时延
        f_d = np.exp(1 - np.max(np.vstack([np.ones_like(d), d]), axis=0))  # 获得时延收益
        reward = self.service_w[req_success_info[:, 0].astype(np.int32)] * (
                self.handover_c * (1 - req_success_info[:, 1]) + (1 - self.handover_c) * req_success_info[:, 1]) * f_d
        reward = np.sum(reward)   # 取平均

        # 状态更新
        # #============ 计算 u =====================
        # # 动作映射 先选3和2 避免混淆
        count = np.sum(action == 3)
        action[action == 3] = np.random.choice(np.array(avail[2]), count, replace=False)  # 不放回选取
        count = np.sum(action == 2)
        action[action == 2] = np.random.choice(np.array(avail[1]), count, replace=False)  # 不放回选取
        count = np.sum(action == 1)
        action[action == 1] = np.random.choice(np.array(avail[0]), count, replace=False)  # 不放回选取
        # # 筛选有效动作
        action_valid = action[action != 0]  # 选择非0的动作值
        action_valid = action_valid - 1  # 转换被选择的DC编号
        # # 筛选获得DC时隙的请求信息
        req_valid_success = req_success_info[action != 0]  # 筛选成功接入请求的信息
        timer = np.ceil(d).astype(np.int32)  # 根据时延 产生倒数计时器
        req_valid_success[:, 2] = timer[action != 0]
        # # 信道利用信息
        self.cum[t, action_valid] = req_valid_success

        if t == self.T - 1:  # eps结束
            done = True
            next_state = None
            next_avail = None
        else:
            done = False
            # # 时间更新
            self.cum[t + 1] = self.cum[t]
            self.cum[t + 1, :, 2][self.cum[t, :, 2] > 0] = self.cum[t, :, 2][self.cum[t, :, 2] > 0] - 1  # 计时器更新
            # # 释放DC
            self.cum[t + 1, self.cum[t + 1][:, 2] == 0, 0] = 6  # 计时器为0信息清空 设置为st为 类别6
            self.cum[t + 1, self.cum[t + 1][:, 2] == 0, 1] = 2  # 计时器为0信息清空 设置为h为 类别2
            # # 得到 u
            cum_w_old = state[0]  #
            cum_w_new = np.append(cum_w_old[1:], self.cum[t + 1:t + 2], axis=0)  # 先入先出
            # # ==============处理请求，m=====================
            req_list = self.req_list[self.req_list['Subframe'] == t+1]  # 选择初始子帧数据
            req_array = np.array(req_list)  # 转换array
            req_info = req_array[:, [2, 3, 5]]  # 选取有效信息作为 m^t_i  [index,st,h,packet,packet_norm]
            # #随机接入过程
            success_idx = self.rac(req_array.shape[0])  # 接入成功索引
            req_success_info = req_info[success_idx]  # 接入请求信息
            assert req_success_info.shape[0] > 0, 'no req'
            # # ===================TDD=========================
            g_delta_w_old = state[2]
            g_delta_w_new = np.expand_dims(np.tile(self.g_delta[t + 1:t + 2], (1, self.dc_num)), axis=2)  # 复制dc_num次
            g_delta_w_new = np.append(g_delta_w_old[1:], g_delta_w_new, axis=0)  # TDD变量 与信道利用矩阵对应
            next_state = (cum_w_new, req_success_info, g_delta_w_new)
            # avail
            avail_dc = np.arange(self.dc_num)[self.cum[t + 1, :, 2] == 0]  # 未被占用的DC
            avail1 = (avail_dc[avail_dc < 8] + 1).tolist()
            avail2 = (avail_dc[(avail_dc > 7) & (avail_dc < 12)] + 1).tolist()
            avail3 = (avail_dc[avail_dc > 11] + 1).tolist()
            next_avail = (avail1, avail2, avail3)

        return reward, next_state, next_avail, done


# if __name__ == "__main__":
#
#     env = Env('data_1', 'dqn', 10)
#     state = env.reset()