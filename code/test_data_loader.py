import matplotlib
from data_loader import *


# ========== PARAMS ===========
BATCH_SIZE = 64


# ========== LOADER INITIALIZATION =============
# available dataset name : COCO, f8k, f30k
# available data type name : train, val, test
# 	'test' type is not available for COCO currently


# ==============================================
loader = DataLoader(batch_size = 64)
loader.create_batches(file_path = '..', batch_size = BATCH_SIZE, dataset_title = 'f8k', 
    data_type = 'test', vocab_size = int(1e4), max_len = 30, skip = None)


# ========= SOME FIELDS ===========

print('total samples {}'.format(loader.count))
print('num_batches {}'.format(loader.num_batches))
print('vocab_size {}'.format(loader.vocab_size))
print('max_sentence_len {}'.format(loader.max_len))


# ======== TEST PROCESS ===========
num_batches = 0
while loader.has_next_batch():
    imgs, caps = loader.next_batch(True)
    print(imgs[0])
    input()
    idx = np.random.randint(0, BATCH_SIZE)
    print(caps[idx])
    cap_str = [loader.idx2word[c] for c in caps[idx]]
    print(cap_str)
    # this does not work well since the mean value is substracted from image
    # if only want to test bgr format, ignore about line 91 in utils_coco.py, img = img - mean
    show_img(imgs[idx])
