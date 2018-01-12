from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras import backend as K
import numpy as np
import resnet50_utils
from nn_utils import *
from language_model import *


class ResNet50(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.img_shape = [224, 224, 3]

        with tf.variable_scope('ImageFeatureExtractor'):
            imgs = tf.placeholder(tf.float32, [self.batch_size] + self.img_shape)
            is_train = tf.placeholder(tf.bool)
            input_ker = imgs
            ker = resnet50_utils.ResNet50(include_top=False, input_shape=[224, 224, 3],
                                          weights="imagenet", input_tensor=input_ker, pooling=None)
        _, self.conv_feats = ker.output
        self.conv_feats = tf.reshape(self.conv_feats, [self.batch_size, 196, 1024])
        self.conv_feat_shape = [196, 1024]
        self.imgs = imgs
        self.is_train = is_train
        self.params = [param for param in tf.trainable_variables() if 'ImageFeatureExtractor' in param.name]

    def basic_block(self, input_feats, name1, name2, is_train, apply_bn, c, s=2):
        """ A basic block of ResNets. """
        branch1_feats = convolution_no_bias(input_feats, 1, 1, 4 * c, s, s, name1 + '_branch1')
        branch1_feats = batch_norm(branch1_feats, name2 + '_branch1', is_train, apply_bn, None)

        branch2a_feats = convolution_no_bias(input_feats, 1, 1, c, s, s, name1 + '_branch2a')
        branch2a_feats = batch_norm(branch2a_feats, name2 + '_branch2a', is_train, apply_bn, 'relu')

        branch2b_feats = convolution_no_bias(branch2a_feats, 3, 3, c, 1, 1, name1 + '_branch2b')
        branch2b_feats = batch_norm(branch2b_feats, name2 + '_branch2b', is_train, apply_bn, 'relu')

        branch2c_feats = convolution_no_bias(branch2b_feats, 1, 1, 4 * c, 1, 1, name1 + '_branch2c')
        branch2c_feats = batch_norm(branch2c_feats, name2 + '_branch2c', is_train, apply_bn, None)

        output_feats = branch1_feats + branch2c_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats

    def basic_block2(self, input_feats, name1, name2, is_train, apply_bn, c):
        """ Another basic block of ResNets. """
        branch2a_feats = convolution_no_bias(input_feats, 1, 1, c, 1, 1, name1 + '_branch2a')
        branch2a_feats = batch_norm(branch2a_feats, name2 + '_branch2a', is_train, apply_bn, 'relu')

        branch2b_feats = convolution_no_bias(branch2a_feats, 3, 3, c, 1, 1, name1 + '_branch2b')
        branch2b_feats = batch_norm(branch2b_feats, name2 + '_branch2b', is_train, apply_bn, 'relu')

        branch2c_feats = convolution_no_bias(branch2b_feats, 1, 1, 4 * c, 1, 1, name1 + '_branch2c')
        branch2c_feats = batch_norm(branch2c_feats, name2 + '_branch2c', is_train, apply_bn, None)

        output_feats = input_feats + branch2c_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats


class HACap(object):
    def __init__(self, data_loader):
        self.d_params = []
        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size
        self.ResNet50 = ResNet50(self.batch_size)
        self.sequence_length = data_loader.max_len
        self.goal_size = 16
        self.step_size = 4
        self.rollout_num = 6
        self.f_params = self.ResNet50.params
        self.text_dropout = 0.75
        dis_embedding_dim = 128
        dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, self.sequence_length]
        dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
        hidden_dim = 128
        GOAL_OUT_SIZE = sum(dis_num_filters)

        self.discriminator = Discriminator(data_loader.max_len, num_classes=2,
                                           vocab_size=data_loader.vocab_size, dis_emb_dim=dis_embedding_dim,
                                           filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                           batch_size=data_loader.batch_size, hidden_dim=hidden_dim,
                                           start_token=0, goal_out_size=GOAL_OUT_SIZE, step_size=self.step_size)
        self.d_params = self.discriminator.D_params
        self.generator = LeakGAN(data_loader.max_len, num_classes=2,
                                 vocab_size=data_loader.vocab_size, dis_emb_dim=dis_embedding_dim,
                                 filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                 batch_size=data_loader.batch_size, hidden_dim=hidden_dim, emb_dim=hidden_dim,
                                 start_token=0, goal_out_size=GOAL_OUT_SIZE, step_size=self.step_size,
                                 goal_size=self.goal_size, D_model=self.discriminator, ResNet_model=self.ResNet50)

    def train_via_MLE(self, sess, image_caption):
        image, caption = image_caption

        feed_dict = {self.ResNet50.imgs: image, self.generator.x: caption, self.generator.drop_out: 1.0, K.learning_phase(): 0, self.generator.train: 1.0}

        _ = sess.run(self.generator.pretrain_worker_updates, feed_dict)
        _ = sess.run(self.generator.pretrain_manager_updates, feed_dict)

    def train_discriminator(self, sess, image_caption):
        image, caption = image_caption

        feed_dict = {self.ResNet50.imgs: image, self.generator.drop_out: 1.0, self.generator.train: 1.0, K.learning_phase(): 0}
        fake_caption = sess.run(self.generator.gen_x, feed_dict)

        label = np.array([[0, 1]] * int(self.batch_size / 2) + [[1, 0]] * int(self.batch_size / 2))
        segment0 = np.concatenate([caption[0:int(self.batch_size / 2)], fake_caption[0:int(self.batch_size / 2)]])
        segment1 = np.concatenate([caption[int(self.batch_size / 2): self.batch_size], fake_caption[int(self.batch_size / 2): self.batch_size]])
        feed_dict_0 = {self.discriminator.D_input_x : segment0, self.discriminator.D_input_y: label, self.discriminator.dropout_keep_prob: 0.75}
        feed_dict_1 = {self.discriminator.D_input_x: segment1, self.discriminator.D_input_y: label, self.discriminator.dropout_keep_prob: 0.75}
        _ = sess.run(self.discriminator.D_train_op, feed_dict_0)
        _ = sess.run(self.discriminator.D_train_op, feed_dict_1)

    def get_reward(self, gen, dis, sess, input_x, img, rollout_num, apply_BRA = True):
        rewards = []
        for i in range(rollout_num):
            for given_num in range(1, gen.sequence_length / gen.step_size):
                real_given_num = given_num * gen.step_size
                feed = {gen.x: input_x, gen.given_num: real_given_num, gen.drop_out: 1.0, gen.ResNet_model.imgs: img, K.learning_phase(): 0}
                samples = sess.run(gen.gen_for_reward, feed)
                # print samples.shape
                feed = {dis.D_input_x: samples, dis.dropout_keep_prob: 1.0}
                ypred_for_auc = sess.run(dis.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # the last token reward
            feed = {dis.D_input_x: input_x, dis.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(dis.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[gen.sequence_length / gen.step_size - 1] += ypred
        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        return rewards

    def train_via_reinforce(self, sess, image_caption):
        image, _ = image_caption
        feed_dict = {self.ResNet50.imgs: image, self.generator.drop_out: 1.0, self.generator.train: 1.0, K.learning_phase(): 0}
        caption = sess.run(self.generator.gen_x, feed_dict)

        reward = self.get_reward(self.generator, self.discriminator, sess, caption, image, self.rollout_num, apply_BRA=False)

        feed_dict = {self.ResNet50.imgs: image, self.generator.drop_out: 1.0,
                     self.generator.x: caption, self.generator.reward: reward, K.learning_phase(): 0}

        _ = sess.run(self.generator.manager_updates, feed_dict)
        _ = sess.run(self.generator.worker_updates, feed_dict)

    def generate_caption(self, sess, image_caption, is_train=1.0):
        image, _ = image_caption
        feed_dict = {self.ResNet50.imgs: image, self.generator.drop_out: 1.0, self.generator.train: is_train, K.learning_phase(): 0}
        caption = sess.run(self.generator.gen_x, feed_dict)
        return self.ind_to_str(caption)

    def ind_to_str(self, caption):
        x, y = caption.shape
        captions = []
        for i in range(x):
            cap = []
            for j in range(y):
                cap.append(self.data_loader._idx2word(caption[i, j]))
            captions.append(" ".join(cap))
        return captions

    def get_attention(self, sess, image_caption):
        image, seq = image_caption
        feed_dict = {self.ResNet50.imgs: image, self.generator.x: seq, self.generator.drop_out: 1.0, self.generator.train: 1.0, K.learning_phase(): 0}
        alpha = sess.run(self.generator.alphas, feed_dict)
        return alpha