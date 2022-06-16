custom_imports = dict(
    imports=['src.custom_conv', 'src.custom_init', 'src.custom_cls', 'src.custom_hook']
)

# checkpoint saving
checkpoint_config = dict(interval=20)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
