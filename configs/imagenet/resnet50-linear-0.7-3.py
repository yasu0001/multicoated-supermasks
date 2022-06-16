
# 32 x 8
_base_ = [
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime-mhnn.py'
]

base_dir = './work_dirs/imagenet-res50/'

custom_imports = dict(imports=['src.custom_conv', 'src.custom_init', 'src.custom_cls', 'src.custom_hook'])

sparsity = [0.7,0.9212,0.9685]

grad_type = 'STE'

# This parameter is 1.0 in all experiments
bw_scale = 1.0

exp_name = f'linear-example'

conv_cfg= dict(
    type = 'MultiLevelHN',
    sparsity=sparsity,
    grad_type=grad_type,
    bw_scale=bw_scale,
)

fc_cfg=dict(
    type='CustomClsHead',
    num_classes=1000,
    in_channels=2048,
    sparsity=sparsity,
    grad_type=grad_type,
    bw_scale=bw_scale,
    loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    topk=(1,5),
    init_cfg=[
        dict(type='Sign', layer='CustomLinear'),
        dict(type='ScoreKaiming', layer='CustomLinear')
    ]
)

custom_hooks = [
    dict(
        type = 'UpdateKthValuesHook'
    ),
    dict(
        type='SaveBestPrec', base_dir=base_dir, name=exp_name ,priority=71
    ),
    # dict(
    #     type='SetCustomScaleFromTextHook', path='configs/resnet18-cifar10/scales/learned-0.7.csv', scale=1.0
    # )
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style='pytorch',
        # cfg can add custom parameter
        conv_cfg=conv_cfg,
        init_cfg=[
            dict(
                type='Sign', 
                layer=['MultiLevelHN']
            ),
            dict(
                type='Constant',
                val=1,
                layer=['_BatchNorm', 'GroupNorm']
            ),
            dict(
                type='ScoreKaiming',
                layer=['MultiLevelHN']
            )
        ]
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=fc_cfg
)


runner = dict(type='EpochBasedRunner', max_epochs=1)
work_dir = f'{base_dir}{exp_name}/{len(sparsity)}-{sparsity[0]}'