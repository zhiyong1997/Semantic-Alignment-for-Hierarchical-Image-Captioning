import os
import numpy as np
from utils_coco import ImageLoader, show_img
from test_attention import rgbgrgbgr

def load_data_flickr(file_path, data_type = 'train', skip = None):
	img_folder = os.path.join(file_path, 'image')


	# == image file names in the datasets ==
	img_file_names = []

	count = 0
	with open(os.path.join(file_path, data_type + '.txt'), 'r') as f:
		for line in f:
			if skip is not None and skip is not 0:
				if count % skip == 0:
					img_file_names.append(line.strip())
			else:
				img_file_names.append(line.strip())
			count += 1

	# ===== images =======
	N = len(img_file_names)
	imgs = np.zeros(shape = [N, 224, 224, 3], dtype = np.float32)

	image_loader = ImageLoader()
	for i in range(N):
		if i % 500 == 0:
			print('loading ' + file_path + ' ' + data_type + ' image {}/{}'.format(i, N))
		img = image_loader.load_img(os.path.join(img_folder, img_file_names[i]))
		imgs[i] = img

	# ===== caps =======
	print('reading dict from disk...')
	caps_dict = {}
	with open(os.path.join(file_path, 'dict.txt')) as f:
		for line in f:
			img_name, cap = _process_flickr_cap(line.strip())
			if not img_name in caps_dict:
				caps_dict[img_name] = []
			caps_dict[img_name].append(cap)				
	
	caps = []
	for img_file_name in img_file_names:
		caps.append(caps_dict.get(img_file_name))

	return imgs, np.array(caps)

def _process_flickr_cap(string):
	p = string.find('#')
	img_name, cap = string[:p], string[p+3:]
	return img_name, cap

	


if __name__ == '__main__':
	load_data_8k('../f8k')