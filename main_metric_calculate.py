import os
import random
import yaml
import cv2
import numpy as np
import argparse
import torch.utils.data
import torchvision.transforms as transforms
import metrics.lpips_fs as lpips

from tqdm import tqdm
from PIL import Image
from configs.default_configs import dataset_configs


def FID(real, fake):
    print('Calculating FID...')
    print('real dir: {}'.format(real))
    print('fake dir: {}'.format(fake))
    command = 'python -m pytorch_fid {} {}'.format(real, fake)
    return os.system(command)

def LPIPS(root):
    print('Calculating LPIPS...')
    print()
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    model = loss_fn_vgg
    model.cuda()

    files = os.listdir(root)
    data = {}
    for file in tqdm(files, desc='loading data'):
        cls = file.split('_')[0]
        idx = int(file.split('_')[1][:-4])
        img = lpips.im2tensor(cv2.resize(lpips.load_image(os.path.join(root, file)), (32, 32)))
        data.setdefault(cls, {})[idx] = img

    classes = set([file.split('_')[0] for file in files])
    res = []
    for cls in tqdm(classes):
        data_temp = torch.cat(list(data[cls].values()), dim=0).cuda()
        output = model(data_temp, normalize=True)
        res.append(output)
    print(np.mean(res))

def unloader(img):
    img = ((img + 1) / 2).clamp(0, 1)
    tf = transforms.Compose([
        transforms.ToPILImage()
    ])
    return tf(img)

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def prepare_real(real_dir, ds_name):
    data = np.load(dataset_configs[ds_name]['data_path'])
    print(data.shape)
    if ds_name == 'flowers':
        data = data[85:]
        num = 10
    elif ds_name == 'animalfaces':
        data = data[119:]
        num = 10
    elif ds_name == 'vggfaces':
        data = data[1802:]
        num = 30
    data_for_fid = data[:, num:, :, :, :]
    os.makedirs(real_dir)
    for cls in tqdm(range(data_for_fid.shape[0]), desc='preparing real images'):
        for idx in range(data_for_fid.shape[1]):
            real_img = data_for_fid[cls, idx, :, :, :]
            if ds_name == 'vggfaces':
                real_img *= 255
            real_img = Image.fromarray(np.uint8(real_img))
            real_img.save(os.path.join(real_dir, '{}_{}.png'.format(cls, str(idx).zfill(3))), 'png')

if __name__ == '__main__':
    
    '''
    Calculating the main metrics.

    Template:

    python main_metric_calculate.py \\
        --real_dir <real_directory> \\
        --fake_dir <fake_directory> \\
        --dataset_name <dataset_name>

    Examples:

    Retrieve test images out of npy files for the first run. The test images will be stored in "real_dir".

    python main_metric_calculate.py \\
        --real_dir ./test/flowers \\
        --fake_dir output/00000-flowers_1_shot/few-shot_samples/magnitude-1.0 \\
        --dataset_name flowers

    python main_metric_calculate.py \\
        --real_dir ./test/animalfaces \\
        --fake_dir output/00001-animalfaces_1_shot/few-shot_samples/magnitude-0.9 \\
        --dataset_name animalfaces

    python main_metric_calculate.py \\
        --real_dir ./test/animalfaces \\
        --fake_dir output/00001-vggfaces_1_shot/few-shot_samples/magnitude-0.9 \\
        --dataset_name vggfaces
    
    Later evaluation DO NOT need to specify the argument "--dataset_name"

    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dir', type=str, help='Directory of the real images for calculating FID.', required=True)
    parser.add_argument('--fake_dir', type=str, help='Directory of the generated images.', required=True)
    parser.add_argument('--dataset_name', help='Name of dataset, need to be specifed when real dir does not exist.', type=str, default=None)
    parser.add_argument('--mode', type=int, help='Evaluate FID / LPIPS, -1 for both, 0 for FID only, 1 for LPIPS only.', default=-1, required=False)
    args = parser.parse_args()
    args.gpu = os.environ['CUDA_VISIBLE_DEVICES']
    
    if not os.path.exists(args.real_dir):
        assert args.dataset_name in ['flowers', 'animalfaces', 'vggfaces']
        print('Preparing real datasets')
        prepare_real(args.real_dir, args.dataset_name)

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    real_dir = args.real_dir
    fake_dir = args.fake_dir
    print('real dir: ', real_dir)
    print('fake dir: ', fake_dir)

    if args.mode == -1 or args.mode == 0:
        if FID(real_dir, fake_dir) != 0:
            exit()
    if args.mode == -1 or args.mode == 1:
        LPIPS(fake_dir)