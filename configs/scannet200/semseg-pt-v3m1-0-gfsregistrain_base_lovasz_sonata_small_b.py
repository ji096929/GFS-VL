weight = ''
resume = False
evaluate = False
test_only = False
seed = 42
save_path = './exp/scannet200/base_lovasz_s42'
num_worker = 4
batch_size = 12
batch_size_val = None
batch_size_test = None
epoch = 20
eval_epoch = 20
sync_bn = False
enable_amp = True
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
mix_prob = 0
param_dicts = [dict(keyword='backbone', lr=0.0001)]
hooks = [
    dict(type='CheckpointLoader', strict=False),
    dict(type='BackboneLoader', strict=False),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='CheckpointSaver_MultiRegis', save_freq=3),
    dict(type='GFS_MultiregTrain_PreciseEvaluator', test_last=True)
]
train = dict(type='GFS_VL_Trainer')
test = dict(type='GFSSemSegTester', verbose=True)
CLASS_LABELS_BASE_NOVEL = [
    'floor', 'wall', 'door', 'chair', 'window', 'trash can', 'table',
    'ceiling', 'doorframe', 'cabinet', 'object', 'shelf', 'desk', 'sink',
    'picture', 'backpack', 'couch', 'box', 'pillow', 'bed', 'radiator',
    'mirror', 'whiteboard', 'lamp', 'curtain', 'toilet', 'book', 'monitor',
    'bookshelf', 'towel', 'tv', 'clothes', 'coffee table', 'office chair',
    'nightstand', 'bag', 'dresser', 'toilet paper', 'recycling bin',
    'kitchen cabinet', 'refrigerator', 'bathtub', 'telephone', 'plant',
    'stool', 'keyboard', 'shoe', 'jacket', 'shower curtain', 'armchair',
    'microwave', 'computer tower', 'bathroom vanity', 'kitchen counter',
    'shower wall', 'paper towel dispenser', 'file cabinet'
]
CLASS_LABELS_BASE = ('refrigerator', 'desk', 'curtain', 'bookshelf', 'bed',
                     'table', 'window', 'cabinet', 'door', 'chair', 'floor',
                     'wall')
vis = None
ps_thresh = 0.6
ai_thresh = 0.9
nb_mix_blks = 3
vlm_3d_weight = './pretrain_weights/sparseunet32_636.pth'
backbone_weight = './exp/scannet200/ptv3_bk/model/model_best.pth'
model = dict(
    type='RegisTrainSegmentor',
    freeze_enc=False,  # Stage-2B：encoder 全解冻
    num_base_classes=12,
    num_novel_classes=45,
    backbone_out_channels=64,
    backbone=dict(
        type='PT-v3m1',
        in_channels=9,
        embedding_type='linear',
        norm_all_ln=True,
        order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=('ScanNet', 'S3DIS', 'Structured3D')),
    criteria=[dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1)])
optimizer = dict(type='Adam', lr=0.001)
scheduler = dict(type='MultiStepLR', milestones=[50, 100], gamma=0.5)
use_lovasz = True
data_root = '/data01/jh/data/scannet200_gfs'
k_shot = 1
regis_train_list = ['regis1']
data = dict(
    num_bases=12,
    num_base_novels=57,
    ignore_index=-1,
    names=[
        'floor', 'wall', 'door', 'chair', 'window', 'trash can', 'table',
        'ceiling', 'doorframe', 'cabinet', 'object', 'shelf', 'desk', 'sink',
        'picture', 'backpack', 'couch', 'box', 'pillow', 'bed', 'radiator',
        'mirror', 'whiteboard', 'lamp', 'curtain', 'toilet', 'book', 'monitor',
        'bookshelf', 'towel', 'tv', 'clothes', 'coffee table', 'office chair',
        'nightstand', 'bag', 'dresser', 'toilet paper', 'recycling bin',
        'kitchen cabinet', 'refrigerator', 'bathtub', 'telephone', 'plant',
        'stool', 'keyboard', 'shoe', 'jacket', 'shower curtain', 'armchair',
        'microwave', 'computer tower', 'bathroom vanity', 'kitchen counter',
        'shower wall', 'paper towel dispenser', 'file cabinet'
    ],
    train=dict(
        type='ScanNet200Dataset_REGISTrain',
        split='train',
        data_root='/data01/jh/data/scannet200_gfs',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='RandomDropout',
                dropout_ratio=0.2,
                dropout_application_ratio=0.2),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='x',
                p=0.5),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='y',
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(
                type='ElasticDistortion',
                distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type='ChromaticAutoContrast', p=0.2, blend_factor=None),
            dict(type='ChromaticTranslation', p=0.95, ratio=0.05),
            dict(type='ChromaticJitter', p=0.95, std=0.05),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='SphereCrop', point_max=65536, mode='random'),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'color', 'mask'),
                feat_keys=('coord', 'color', 'normal'))
        ],
        test_mode=False,
        loop=1),
    regis1=dict(
        type='ScanNet200Dataset_REGIS',
        split='train',
        seed=10,
        k_shot=1,
        data_root='/data01/jh/data/scannet200_gfs',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'color', 'name'),
                feat_keys=('coord', 'color', 'normal'))
        ],
        test_mode=False),
    regis2=dict(
        type='ScanNet200Dataset_REGIS',
        split='train',
        seed=20,
        k_shot=1,
        data_root='/data01/jh/data/scannet200_gfs',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'color', 'name'),
                feat_keys=('coord', 'color', 'normal'))
        ],
        test_mode=False),
    regis3=dict(
        type='ScanNet200Dataset_REGIS',
        split='train',
        seed=30,
        k_shot=1,
        data_root='/data01/jh/data/scannet200_gfs',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'color', 'name'),
                feat_keys=('coord', 'color', 'normal'))
        ],
        test_mode=False),
    regis4=dict(
        type='ScanNet200Dataset_REGIS',
        split='train',
        seed=40,
        k_shot=1,
        data_root='/data01/jh/data/scannet200_gfs',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'color', 'name'),
                feat_keys=('coord', 'color', 'normal'))
        ],
        test_mode=False),
    regis5=dict(
        type='ScanNet200Dataset_REGIS',
        split='train',
        seed=50,
        k_shot=1,
        data_root='/data01/jh/data/scannet200_gfs',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'color', 'name'),
                feat_keys=('coord', 'color', 'normal'))
        ],
        test_mode=False),
    val=dict(
        type='ScanNet200Dataset_TEST',
        split='val',
        data_root='/data01/jh/data/scannet200_gfs',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'color'),
                feat_keys=('coord', 'color', 'normal'))
        ],
        test_mode=False),
    test=dict(
        type='ScanNet200Dataset_TEST',
        split='val',
        data_root='/data01/jh/data/scannet200_gfs',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='NormalizeColor')
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='test',
                keys=('coord', 'color', 'normal'),
                return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type='CenterShift', apply_z=False),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index', 'color'),
                    feat_keys=('coord', 'color', 'normal'))
            ],
            aug_transform=[[{
                'type': 'RandomRotateTargetAngle',
                'angle': [0],
                'axis': 'z',
                'center': [0, 0, 0],
                'p': 1
            }]])))
