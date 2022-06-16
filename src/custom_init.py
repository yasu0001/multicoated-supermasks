from mmcv.cnn import INITIALIZERS 
from mmcv.cnn.utils.weight_init import BaseInit, _get_bases_name, update_init_info
import torch
import torch.nn as nn
import math

def sign_init(module, init_scale):
    if hasattr(module, 'weight') and module.weight is not None:
        fan = nn.init._calculate_correct_fan(module.weight, 'fan_in')
        scale_fan = fan * (1 - module.sparsity[0])
        gain = nn.init.calculate_gain('relu')
        std = gain / math.sqrt(scale_fan)
        std *= init_scale

        module.weight.data = module.weight.data.sign()
        module.scale.data = torch.Tensor([std])

def score_kaiming_init(module,
                       a = 0,
                       mode='fan_out',
                       nonlinearity='relu',
                       bias=0,
                       distribution='normal'):
    assert distribution in ['uniform', 'normal']

    if hasattr(module, 'score') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(module.score, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.score, a=a, mode=mode, nonlinearity=nonlinearity)

    if hasattr(module, 'scores') and module.weight is not None:
        with torch.no_grad():
            if distribution == 'uniform':
                for i in range(len(module.scores)):
                    nn.init.kaiming_uniform_(module.scores[i], a=a, mode=mode, nonlinearity=nonlinearity)
            else:
                for i in range(len(module.scores)):
                    nn.init.kaiming_normal_(module.scores[i], a=a, mode=mode, nonlinearity=nonlinearity)

@INITIALIZERS.register_module(name='Sign')
class SignInit(BaseInit):
    def __init__(self, init_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.init_scale = init_scale

    def __call__(self, module):

        def init(m):
            if self.wholemodule:
                sign_init(m)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    sign_init(m, self.init_scale)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}'
        return info         

@INITIALIZERS.register_module(name='ScoreKaiming')
class ScoreKaimingInit(BaseInit):
    def __init__(self,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 distribution='normal',
                 **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.distribution = distribution

    def __call__(self, module):

        def init(m):
            if self.wholemodule:
                score_kaiming_init(m, self.a, self.mode, self.nonlinearity,
                             self.bias, self.distribution)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    score_kaiming_init(m, self.a, self.mode, self.nonlinearity,
                                 self.bias, self.distribution)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: a={self.a}, mode={self.mode}, ' \
               f'nonlinearity={self.nonlinearity}, ' \
               f'distribution ={self.distribution}, bias={self.bias}'
        return info
