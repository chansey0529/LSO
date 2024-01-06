DATA_PATH = '../datasets_npy'
CKPT_PATH = '../Conditional-StyleGAN_seen-ckpt'
WS_PATH = '../II2S/outputs/inversion'
IDCKPT_PATH = 'pretrained_models'

'''

Set the following paths before running.

    - DATA_PATH
        - animal_128.npy
        - flower_c8189_s128_data_rgb.npy
        - vgg_face_data_rgb.npy

    - CKPT_PATH
        - flowers
            - network-snapshot-004838.pkl
        - animalfaces
            - network-snapshot-021600.pkl
        - vggfaces
            - network-snapshot-022579.pkl

    - WS_PATH
        - flowers_w
            - flowers_unseen17_0-10_step1300.npy
        - animals_w
            - animalfaces_unseen30_0-10_step1300.npy
        - vggfaces_w
            - vggfaces_unseen552_0-30_step1300.npy

    - IDCKPT_PATH
        - model_ir_se50.pth
        - moco_v2_800ep_pretrain.pt

'''

dataset_configs = {
    'flowers': {
        'n_seen': 85,
        'n_unseen': 17,
        'n_unseen_samples': 10,
        'stylegan_weights': f'{CKPT_PATH}/flowers/network-snapshot-004838.pkl',
        'data_path': f'{DATA_PATH}/flower_c8189_s128_data_rgb.npy',
        'test_ws_path': f'{WS_PATH}/flowers_w/flowers_unseen17_0-10_step1300.npy',
        'resolution': 128,
    },
    'animalfaces': {
        'n_seen': 119,
        'n_unseen': 30,
        'n_unseen_samples': 10,
        'stylegan_weights': f'{CKPT_PATH}/animalfaces/network-snapshot-021600.pkl',
        'data_path': f'{DATA_PATH}/animal_128.npy',
        'test_ws_path': f'{WS_PATH}/animals_w/animalfaces_unseen30_0-10_step1300.npy',
        'resolution': 128,
    },
    'vggfaces': {
        'n_seen': 1802,
        'n_unseen': 552,
        'n_unseen_samples': 30,
        'stylegan_weights': f'{CKPT_PATH}/vggfaces/network-snapshot-022579.pkl',
        'data_path': f'{DATA_PATH}/vgg_face_data_rgb.npy',
        'test_ws_path': f'{WS_PATH}/vggfaces_w/vggfaces_unseen552_0-30_step1300.npy',
        'resolution': 64,
    }
}

path_configs = {
    'id_path': f'{IDCKPT_PATH}/model_ir_se50.pth',
    'moco_path': f'{IDCKPT_PATH}/moco_v2_800ep_pretrain.pt',
}
