from mmcv.runner.hooks import HOOKS, Hook
from .custom_conv import MultiLevelHN
from mmcv.cnn import Conv2d
import numpy as np
import torch
import math

@HOOKS.register_module()
class UpdateKthValuesHook(Hook):
    def __init__(self):
        pass

    def before_run(self, runner):
        runner.logger.info(f'Set kth values in each layer')
        self._set_kth_values(runner.model)
        runner.logger.info(f'Complete setting kth values in each layer')

    def after_train_iter(self, runner):
        self._set_kth_values(runner.model)
    
    def _set_kth_values(self, module):
        for n, m in module.named_modules():
            if hasattr(m, 'set_kthvalues'):
                m.set_kthvalues()

@HOOKS.register_module()
class SaveBestPrec(Hook):
    def __init__(self, base_dir='./work_dir', name='test'):
        self.best_top1 = 0.
        self.best_top5 = 0.
        self.base_dir = base_dir
        self.name = name

    def after_train_epoch(self, runner):
        # cul_top1 = runner.log_buffer.output['accuracy_top-1']
        # cul_top5 = runner.log_buffer.output['accuracy_top-5']
        cul_top1 = runner.log_buffer.output.get('accuracy_top-1')
        cul_top5 = runner.log_buffer.output.get('accuracy_top-5')

        if cul_top1 is not None:
            if self.best_top1 < cul_top1:
                self.best_top1 = cul_top1
                self.best_top5 = cul_top5
        else:
            print(f'Warning : {runner.log_buffer.output}')
    
    def after_run(self, runner):
        quant_num, prune_rate, params = self._get_conv_info(runner)
        self._write_results_to_csv(
            base_dir=self.base_dir,
            now=runner.timestamp,
            name=self.name,
            quant_num=quant_num,
            prune_rate=prune_rate,
            params=params,
            best_acc1=self.best_top1,
            best_acc5=self.best_top5
        )

    def _get_conv_info(self, runner):
        quant_num = None
        prune_rate = None
        params = calc_model_size(runner.model)
        for n, m in runner.model.named_modules():
            if isinstance(m, Conv2d):
                quant_num = None
                prune_rate = None
                break
            elif isinstance(m, MultiLevelHN):
                quant_num = len(m.sparsity)
                prune_rate = m.sparsity[0]
                break
            
        return quant_num, prune_rate, params

    def _write_results_to_csv(self, base_dir ,**kwargs):
        import pathlib
        results = pathlib.Path(base_dir) / "results.csv"

        if not results.exists():
            results.write_text(
                "Data,"
                "Name,"
                "Quant,"
                "Pruning,"
                "Size(MB),"
                "BestAcc1,"
                "BestAcc5\n"
            )
        
        with open(results, 'a+') as f:
            f.write(
                (
                    "{now},"
                    "{name},"
                    "{quant_num},"
                    "{prune_rate},"
                    "{params},"
                    "{best_acc1:.02f},"
                    "{best_acc5:.02f}\n"
                ).format(**kwargs)
            )

def calc_model_size(model):
    params = 0.
    for n,m in model.named_modules():
        if hasattr(m, 'weight') and hasattr(m.weight, "size") and m.weight.requires_grad:
            params += m.weight.numel() * 32
        if hasattr(m, 'score') and hasattr(m.score, "size") and m.score.requires_grad:
            params += m.score.numel()
            if len(m.sparsity) == 1:
                continue
            for i in range(len(m.sparsity)-1):
                # print(m.score.numel() * (1 - m.sparsity[i]))
                params += math.ceil(m.score.numel() * (1 - m.sparsity[i]))


        if hasattr(m, 'bias') and hasattr(m.bias, "size") and m.bias.requires_grad:
            #params += m.bias.numel()
            pass
    
    params = params / (8 * 1024 ** 2)
    return params