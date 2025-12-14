import abc
import math
from abc import ABC
import numpy as np
import random
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from einops import rearrange, repeat
from surrogate import *


class BaseNode(nn.Module, abc.ABC):
    """
    神经元模型的基类
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param mem_detach: 是否将上一时刻的膜电位在计算图中截断
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self,
                 threshold=.5,
                 v_reset=0.,
                 dt=1.,
                 step=8,
                 requires_thres_grad=False,
                 sigmoid_thres=False,
                 requires_fp=False,
                 layer_by_layer=False,
                 n_groups=1,
                 *args,
                 **kwargs):

        super(BaseNode, self).__init__()
        self.threshold = Parameter(torch.tensor(threshold), requires_grad=requires_thres_grad)
        self.sigmoid_thres = sigmoid_thres
        self.mem = 0.
        self.spike = 0.
        self.dt = dt
        self.feature_map = []
        self.mem_collect = []
        self.requires_fp = requires_fp
        self.v_reset = v_reset
        self.step = step
        self.layer_by_layer = layer_by_layer
        self.groups = n_groups
        self.mem_detach = kwargs['mem_detach'] if 'mem_detach' in kwargs else False
        self.requires_mem = kwargs['requires_mem'] if 'requires_mem' in kwargs else False

    @abc.abstractmethod
    def calc_spike(self):
        """
        通过当前的mem计算是否发放脉冲, 并reset
        :return: None
        """

        pass

    def integral(self, inputs):
        """
        计算由当前inputs对于膜电势的累积
        :param inputs: 当前突触输入电流
        :type inputs: torch.tensor
        :return: None
        """

        pass

    def get_thres(self):
        return self.threshold if not self.sigmoid_thres else self.threshold.sigmoid()

    def rearrange2node(self, inputs):
        if self.groups != 1:
            if len(inputs.shape) == 4:
                outputs = rearrange(inputs, 'b (c t) w h -> t b c w h', t=self.step)
            elif len(inputs.shape) == 2:
                outputs = rearrange(inputs, 'b (c t) -> t b c', t=self.step)
            else:
                raise NotImplementedError

        elif self.layer_by_layer:
            if len(inputs.shape) == 4:
                outputs = rearrange(inputs, '(t b) c w h -> t b c w h', t=self.step)
            elif len(inputs.shape) == 2:
                outputs = rearrange(inputs, '(t b) c -> t b c', t=self.step)
            else:
                raise NotImplementedError


        else:
            outputs = inputs

        return outputs

    def rearrange2op(self, inputs):
        if self.groups != 1:
            if len(inputs.shape) == 5:
                outputs = rearrange(inputs, 't b c w h -> b (c t) w h')
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs, ' t b c -> b (c t)')
            else:
                raise NotImplementedError
        elif self.layer_by_layer:
            if len(inputs.shape) == 5:
                outputs = rearrange(inputs, 't b c w h -> (t b) c w h')
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs, ' t b c -> (t b) c')
            else:
                raise NotImplementedError

        else:
            outputs = inputs

        return outputs

    def forward(self, inputs):
        """
        torch.nn.Module 默认调用的函数，用于计算膜电位的输入和脉冲的输出
        在```self.requires_fp is True``` 的情况下，可以使得```self.feature_map```用于记录trace
        :param inputs: 当前输入的膜电位
        :return: 输出的脉冲
        """

        if self.layer_by_layer or self.groups != 1:
            inputs = self.rearrange2node(inputs)

            outputs = []
            for i in range(self.step):
                
                if self.mem_detach and hasattr(self.mem, 'detach'):
                    self.mem = self.mem.detach()
                    self.spike = self.spike.detach()
                self.integral(inputs[i])
                
                self.calc_spike()
                
                if self.requires_fp is True:
                    self.feature_map.append(self.spike)
                if self.requires_mem is True:
                    self.mem_collect.append(self.mem)
                outputs.append(self.spike)
            outputs = torch.stack(outputs)

            outputs = self.rearrange2op(outputs)
            return outputs
        else:
            if self.mem_detach and hasattr(self.mem, 'detach'):
                self.mem = self.mem.detach()
                self.spike = self.spike.detach()
            self.integral(inputs)
            self.calc_spike()
            if self.requires_fp is True:
                self.feature_map.append(self.spike)
            if self.requires_mem is True:
                self.mem_collect.append(self.mem)   
            return self.spike

    def n_reset(self):
        """
        神经元重置，用于模型接受两个不相关输入之间，重置神经元所有的状态
        :return: None
        """
        # print("LIF node reset!!!")
        self.mem = self.v_reset
        self.spike = 0.
        self.feature_map = []
        self.mem_collect = []
    def get_n_attr(self, attr):

        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            return None

    def set_n_warm_up(self, flag):
        """
        一些训练策略会在初始的一些epoch，将神经元视作ANN的激活函数训练，此为设置是否使用该方法训练
        :param flag: True：神经元变为激活函数， False：不变
        :return: None
        """
        self.warm_up = flag

    def set_n_threshold(self, thresh):
        """
        动态设置神经元的阈值
        :param thresh: 阈值
        :return:
        """
        self.threshold = Parameter(torch.tensor(thresh, dtype=torch.float), requires_grad=False)

    def set_n_tau(self, tau):
        """
        动态设置神经元的衰减系数，用于带Leaky的神经元
        :param tau: 衰减系数
        :return:
        """
        if hasattr(self, 'tau'):
            self.tau = Parameter(torch.tensor(tau, dtype=torch.float), requires_grad=False)
        else:
            raise NotImplementedError

#============================================================================
# node的基类
class BaseMCNode(nn.Module, abc.ABC):
    """
    多房室神经元模型的基类
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param comps: 神经元不同房室, 例如["apical", "basal", "soma"]
    """
    def __init__(self,
                 threshold=1.0,
                 v_reset=0.,
                 comps=[]):
        super().__init__()
        self.threshold = Parameter(torch.tensor(threshold), requires_grad=False)
        # self.decay = Parameter(torch.tensor(decay), requires_grad=False)
        self.v_reset = v_reset
        assert len(comps) != 0
        self.mems = dict()
        for c in comps:
            self.mems[c] = None 
        self.spike = None
        self.warm_up = False

    @abc.abstractmethod
    def calc_spike(self):
        pass
    @abc.abstractmethod
    def integral(self, inputs):
        pass        
    
    def forward(self, inputs: dict):
        '''
        Params:
            inputs dict: Inputs for every compartments of neuron 
        '''
        if self.warm_up:
            return inputs
        else:
            self.integral(**inputs)
            self.calc_spike()
            return self.spike

    def n_reset(self):
        # print("MCNode Reset!!!")
        for c in self.mems.keys():
            self.mems[c] = self.v_reset
        self.spike = 0.0

    def get_n_fire_rate(self):
        if self.spike is None:
            return 0.
        return float((self.spike.detach() >= self.threshold).sum()) / float(np.product(self.spike.shape))

    def set_n_warm_up(self, flag):
        self.warm_up = flag

    def set_n_threshold(self, thresh):
        self.threshold = Parameter(torch.tensor(thresh, dtype=torch.float), requires_grad=False)

class LIFNode(BaseNode):
    """
    Leaky Integrate and Fire
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self, threshold=0.5, tau=2., act_fun=QGateGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        # self.threshold = threshold
        # print(threshold)
        # print(tau)

    def integral(self, inputs):
        self.mem = self.mem + (inputs - self.mem) / self.tau

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach())


class MCNode(BaseMCNode):
    def __init__(self, 
                 threshold=0.5,
                 tau=2.0,
                 tau_basal=2.0,
                 tau_apical=2.0,
                 v_reset=0.0,
                 comps=['basal', 'apical', 'soma'],
                 act_fun=QGateGrad):
        super().__init__(threshold, v_reset, comps)

        self.tau = tau
        self.tau_basal = tau_basal
        self.tau_apical = tau_apical
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=tau, requires_grad=False)

    def integral(self, basal_inputs, apical_inputs, soma_inputs):
        '''
        Params:
            inputs torch.Tensor: Inputs for basal dendrite  
        '''

        self.mems['basal'] =  self.mems['basal'] + (basal_inputs-self.mems['basal'])/ self.tau_basal
        self.mems['apical'] =  self.mems['apical'] + (apical_inputs-self.mems['apical']) / self.tau_apical
        z_t = F.sigmoid(self.mems['apical'])
        h_t = self.mems['basal'] - self.mems['soma'] + soma_inputs
        self.mems['soma'] = self.mems['soma'] + z_t*(h_t-self.mems['soma']) / self.tau


    def calc_spike(self):
        self.spike = self.act_fun(self.mems['soma'] - self.threshold)
        self.mems['soma'] = self.mems['soma']  * (1. - self.spike.detach())
        self.mems['basal'] = self.mems['basal'] * (1. - self.spike.detach())
        self.mems['apical'] = self.mems['apical']  * (1. - self.spike.detach())
