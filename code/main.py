from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from random import random as rdfloat
from HACap import HACap
from utils_coco import ImageLoader
from test_attention import show_sentence_attention
import pickle, os
BATCH_SIZE = 100
skip_rate = 0.0

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('test', True, 'Training or testing a model.')
flags.DEFINE_boolean('init', True, 'To initialize the parameters or to load it from existing file.')
flags.DEFINE_boolean('conti', True, 'To initialize the parameters or to load it from existing file.')
flags.DEFINE_boolean('save', True, 'The training will update the saved model.')
flags.DEFINE_string('dataset', 'f8k', 'The dataset to use.')

model_path = './saved_model/'
dataset_path = {'COCO': 'COCO', 'f8k': 'f8k', 'f30k': 'f30k'}

folder = './attentions/'
if not os.path.exists(folder):
    os.makedirs(folder)

def handle_image_cap_tuple(tup):
    image, caption = tup
    print("\n\n".join(caption[0:4]))


# ==== input ====
# ==== image_cap_tuple : (imgs, caps) get from data_loader.next_batch ====
# ==== attention : (batch_size, sentence_len, 196) ====
def draw_attention(image_cap_tuple, attention, idx2word, batch_index):
    # === select the image and sentence to draw attention ====
    img, att, cap = image_cap_tuple[0][0], attention[0], image_cap_tuple[1][0]
            
    # ==== bgr + mean and to rgb ====
    image_loader = ImageLoader()
    img = image_loader.get_origin_image(img)

    # ==== draw attention ====
    cap = ' '.join([idx2word[c] for c in cap])
    show_sentence_attention(img, att, cap, batch_index)


def plot_line(values, alpha = 1.):
    plt.plot(values, 'r-', alpha = alpha)

# guidance should be (batch_size, 6, sentence_len)
def draw_guidance(image_cap_tuple, guidance, idx2word, batch_index):
    cap, guid = image_cap_tuple[1][0], guidance[0]

    cap = [idx2word[c] for c in cap]
    cap = ((' '.join(cap)).strip()).split(' ')
    max_len = len(cap)

    for i in range(6) : plot_line(guid[i][:max_len])
    plt.axis([0, max_len, 0, 1])
    plt.xticks(np.arange(max_len), cap, rotation = 45)


    folder = os.path.join('guidance', str(batch_index))
    if not os.path.exists(folder):
        os.makedirs(folder)

    # cap is a list without ' ' element, guid is of shape (16, sentence_len)
    with open(os.path.join(folder, 'data'), 'wb') as f:
        pickle.dump((cap, guid), f)
    plt.savefig(os.path.join(folder, 'guidance.pdf'), dpi=300)


def main():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    model_name = model_path + FLAGS.dataset
    dataloader = DataLoader(BATCH_SIZE)
    model = HACap(dataloader)
    saver = tf.train.Saver(tf.global_variables())
    sess.run(tf.global_variables_initializer())
    if FLAGS.test:
        saver.restore(sess, model_name) 
        dataloader.create_batches(file_path = './dataset/', dataset_title = dataset_path[FLAGS.dataset], batch_size = BATCH_SIZE, data_type = "train", skip = 0)
        for i in range(dataloader.num_batches):
            print('batches {} / {}'.format(i, dataloader.num_batches))
            image_cap_tuple = dataloader.next_batch()
            caption = model.generate_caption(sess, image_cap_tuple, is_realized=False)
            image, _ = image_cap_tuple
            image_cap_tuple = image, caption
            draw_attention(image_cap_tuple, 
                np.transpose(model.get_attention(sess, image_cap_tuple), (1, 0, 2)), 
                dataloader.idx2word, i)
        '''   
            draw_guidance(image_cap_tuple, 
                np.transpose(model.get_guidance(sess, image_cap_tuple), (1, 2, 0)))
        '''

        '''
            caption = model.generate_caption(sess, image_caption=image_cap_tuple)
            image, _ = image_cap_tuple
            handle_image_cap_tuple((image, caption))
        '''
    else:
        dataloader.create_batches('./dataset/', BATCH_SIZE, dataset_path[FLAGS.dataset], data_type="train", skip=None)
        print(dataloader.vocab_size)
        if not FLAGS.init:
            saver.restore(sess, model_name)
        else:
            if FLAGS.conti:
                saver.restore(sess, model_name)
            for pretrain_batch in range(10000):
                if pretrain_batch < 5 and (not FLAGS.conti):
                    for pretrain_d_batch in range(5):
                        print("pretraining discriminator epoch#%d-%d" % (pretrain_batch, pretrain_d_batch))
                        dataloader.reset_pointer()
                        for batch_idx in range(dataloader.num_batches):
                            if batch_idx % 1000 == 0:
                                print("pretraining discriminator epoch#%d-%d/%d" % (pretrain_d_batch, batch_idx, dataloader.num_batches))
                            batch = dataloader.next_batch()
                            if rdfloat() > skip_rate:
                                model.train_discriminator(sess, batch)
                for pretrain_g_batch in range(15):
                    print("pretraining generator epoch#%d-%d" % (pretrain_batch, pretrain_g_batch))
                    dataloader.reset_pointer()
                    for batch_idx in range(dataloader.num_batches):
                        batch = dataloader.next_batch()
                        if rdfloat() > skip_rate:
                            model.train_via_MLE(sess, batch)
                        if batch_idx % 1000 == 0:
                            print("pretraining generator epoch#%d-%d/%d" % (pretrain_g_batch, batch_idx, dataloader.num_batches))
                            image_cap_tuple = batch
                            caption = model.generate_caption(sess, image_caption=image_cap_tuple, is_train=0.0)
                            image, truth = image_cap_tuple
                            print("fake:")
                            handle_image_cap_tuple((image, caption))
                            print("real:")
                            handle_image_cap_tuple((image, model.ind_to_str(truth)))
                            print("real_alpha:")
                            print(model.get_attention(sess, image_cap_tuple)[0])
                if FLAGS.save:
                    saver.save(sess, model_name)
        for adv_batch in range(10000000):
            print("adversarial training epoch#%d" % adv_batch)
            dataloader.reset_pointer()
            for batch_idx in range(dataloader.num_batches):
                batch = dataloader.next_batch()
                if rdfloat() > skip_rate:
                    for d_idx in range(2):
                        model.train_discriminator(sess, batch)
                    if adv_batch % 1 == 0:
                        for teacher_forcing_idx in range(5):
                            model.train_via_MLE(sess, batch)
                            if batch_idx % 1000 == 0:
                                image_cap_tuple = batch
                                print(
                                    "adversarial training epoch#%d-%d/%d" % (adv_batch, batch_idx, dataloader.num_batches))
                                caption = model.generate_caption(sess, image_caption=image_cap_tuple)
                                image, truth = image_cap_tuple
                                print("fake:")
                                handle_image_cap_tuple((image, caption))
                                print("real:")
                                handle_image_cap_tuple((image, model.ind_to_str(truth)))

                    for g_idx in range(1):
                        model.train_via_reinforce(sess, batch)
            if FLAGS.save and adv_batch % 5 == 0:
                saver.save(sess, model_name)


if __name__ == "__main__":
    main()
