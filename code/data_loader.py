from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import sys
import os
sys.path.append('cocoapi/PythonAPI')
# from utils_coco import load_data_coco, show_img
from utils_flickr import load_data_flickr

class DataLoader(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.max_len = 20
        self.vocab_size = 5837
        self.idx2word = {}
        self.word2idx = {}
        self.dataset_title ='COCO'
        self.num_batches = 1000

    def create_batches(self, file_path, batch_size, dataset_title, data_type = 'train', vocab_size = 10000, max_len = 20, skip = None):
        self.batch_size = batch_size
        self.dataset_title = dataset_title
        self.data_type = data_type
        self.vocab_size = vocab_size
        self.max_len = max_len

        imgs, caps = self._get_data(file_path, dataset_title, data_type, skip)

        # ====== make word2idx and idx2word =========
        self._make_word_dict(caps) 
        
        # ===== process images and padded captions =====
        self.imgs, self.caps_padded, self.caps_origin = self._process_data(imgs, caps)
        
        # ========= num_whole & num_batches =========
        self.count = len(self.indices)
        self.num_batches = int(self.count * 1.0 / self.batch_size)
        
        # ===== used for select batches ===== 
        self.reset_pointer()
        pass

    def next_batch(self, is_padded=True):
        assert self.has_next_batch()
        start, end = self.current_index, self.current_index + self.batch_size
        current_indices = self.indices[start:end]
        imgs = [self.imgs[idx[0]] for idx in current_indices]
        caps_selected = self.caps_padded if is_padded else self.caps_origin
        caps = [caps_selected[idx[0]][idx[1]] for idx in current_indices]
        self.current_index += self.batch_size
        return np.array(imgs), np.array(caps)
    
    def reset_pointer(self):
        self.current_index = 0
        np.random.shuffle(self.indices)
    
    def has_next_batch(self):
        return self.current_index + self.batch_size <= self.count
        

    def _make_word_dict(self, caps):
        word_count = {}
        for sents in caps:
            for sent in sents:
                sent_split = sent.lower().split(' ')
                for w in sent_split:
                    word_count[w] = word_count.get(w, 0) + 1
        
        sorted_word_count = sorted(list(word_count.items()), key=lambda x: x[1], reverse=True) 
        sorted_word_count = sorted_word_count[:self.vocab_size - 3]
        self.vocab_size = min(self.vocab_size, len(sorted_word_count) + 3)
    
        self.idx2word[0] = '<START_TOKEN>'
        self.idx2word[1] = ' '
        self.idx2word[2] = '<UNKNOWN_TOKEN>'
        self.word2idx['<START_TOKEN>'] = 0
        self.word2idx[' '] = 1
        self.word2idx['<UNKNOWN_TOKEN>'] = 2
        
        for idx in range(self.vocab_size - 3):
            word, freq = sorted_word_count[idx]
            self.idx2word[idx + 3] = word
            self.word2idx[word] = idx + 3
            
    def _process_data(self, imgs, caps):
        num_imgs = len(imgs)
        self.indices = []
        max_idxx = 0
        # caps list of list, cap str
        for idx in range(num_imgs):
            num_captions = len(caps[idx])
            for idxx in range(num_captions):
                max_idxx = max(max_idxx, idxx + 1)
                if (len(caps[idx][idxx].split(' ')) <= self.max_len):
                    self.indices.append((idx, idxx))
        print("Maximum caption number per image:%d" % max_idxx)

        caps_indices = np.zeros(shape=[len(caps), max_idxx, self.max_len], dtype=np.int16)
        for i in range(len(caps)):
            if i % 1000 == 0:
                print("processing captions of image#(%d/%d)" % (i, num_imgs))
            for j in range(len(caps[i])):
                caps_indices[i, j] = np.array(self._str_to_indice(caps[i][j]))
        
        return imgs, caps_indices, caps
        
        
    def _str_to_indice(self, cap):
        words = np.array([self._word2idx(w) for w in cap.lower().split(' ')])
        cap_indice = np.zeros([self.max_len]).astype(np.int16)
        if len(words) <= self.max_len:
            cap_indice[:len(words)] = words
            cap_indice[len(words):] = 1
        return cap_indice


    def _word2idx(self, word):
        if word in self.word2idx.keys():
            return self.word2idx[word]
        else:
            return 2  # <UNKNOWN_TOKEN>

    def _idx2word(self, idx):
        if idx in self.idx2word.keys():
            return self.idx2word[idx]
        else:
            return "<UNK>"  # <UNKNOWN_TOKEN>

    def _get_data(self, file_path, dataset_title, data_type, skip):

        if dataset_title == 'COCO':
            imgs, caps = load_data_coco(os.path.join(file_path, dataset_title), skip = skip)
        elif dataset_title == 'f8k' or dataset_title == 'f30k':
            imgs, caps = load_data_flickr(os.path.join(file_path, dataset_title), 
                data_type = data_type, skip = skip)
        else:
            print(dataset_title)
            print('Not implemented yet for this dataset, now COCO and f8k, f30k are available.')
            pass
            
        return imgs, caps
