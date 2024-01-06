
'''
Custom dataset implementation
Code adapted from following paper
"Training Generative Adversarial Networks with Limited Data."
See LICENSES/LICENSE_STYLEGAN2_ADA.txt for original license.
'''

from torch.utils.data import Dataset
import numpy as np
import torch

#----------------------------------------------------------------------------

class LSODataset(Dataset):
	def __init__(self, image, w, y):
		assert image.shape[0] == w.shape[0]
		assert image.shape[0] == y.shape[0]
		self.image = image # few-shot image [k, 3, img_size, img_size]
		self.w = w # few-shot target anchors (inversed latent code)
		self.y = y # few-shot label [k, n_seen + 1]

	def __len__(self):
		k, _C, _H, _W = self.image.shape
		return k

	def __getitem__(self, index):
		return index, self.image[index], self.w[index], self.y[index]

#----------------------------------------------------------------------------
	
class AllImagesDataset(Dataset):
	def __init__(self, opts):
		self.opts = opts

		all_images = np.load(opts.data_path)
		all_images = torch.tensor(all_images)
		if opts.dataset_name == 'vggfaces':
			all_images *= 255
			all_images = all_images.to(torch.uint8)
		
		all_ws = np.load(opts.test_ws_path)
		
		unseen_image = all_images[opts.n_seen:, :opts.n_unseen_samples, :, :, :]
		nc, n, W, H, C = unseen_image.shape
		self.image = unseen_image.reshape([nc * n, W, H, C])
		self.ws = all_ws

	def __len__(self):
		return self.image.shape[0]

	def __getitem__(self, index):
		ws = torch.tensor(self.ws[index])
		from_img = self.image[index].permute(2, 0, 1) / 127.5 - 1
		return from_img, ws

#----------------------------------------------------------------------------