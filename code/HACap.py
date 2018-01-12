from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.layers import xavier_initializer
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

def SRUCell(params, hidden_dim, input_dim):
    init_matrix = tf.orthogonal_initializer()
    init_vector = xavier_initializer()
    Wr = tf.Variable(init_matrix([input_dim, hidden_dim]))
    br = tf.Variable(init_vector([hidden_dim]))

    Wf = tf.Variable(init_matrix([input_dim, hidden_dim]))
    bf = tf.Variable(init_vector([hidden_dim]))

    Wc = tf.Variable(init_matrix([input_dim, hidden_dim]))
    Wh = tf.Variable(init_matrix([input_dim, hidden_dim]))

    params.extend([
        Wr, br, Wf, bf, Wc, Wh
    ])

    # input_seq: [b x l x i_d]
    def unit(inputs, c_prev):
        f = tf.sigmoid(
            tf.matmul(inputs, Wf) + bf
        )

        r = tf.sigmoid(
            tf.matmul(inputs, Wr) + br
        )

        c = f * c_prev + (1.0 - f) * tf.matmul(inputs, Wc)

        h = r * tf.nn.tanh(c) + (1.0 - r) * tf.matmul(inputs, Wh)

        return c, h
        # input_seq: batch x seq_len x input_dim
    def block(input_seq, c0, batch_size, seq_len):
        xx = tf.transpose(input_seq, perm=[1, 0, 2])
        x = tf.reshape(xx, [-1, input_dim])

        f = tf.reshape(
                tf.sigmoid(
                    tf.matmul(x, Wf) + bf
                ), [seq_len, batch_size, hidden_dim]
        )

        r = tf.reshape(
                tf.sigmoid(
                    tf.matmul(x, Wr) + br
                ), [seq_len, batch_size, hidden_dim]
        )

        cx = tf.reshape(
            tf.matmul(x, Wc), [seq_len, batch_size, hidden_dim]
        )

        hx = tf.reshape(
            tf.matmul(x, Wh), [seq_len, batch_size, hidden_dim]
        )

        c = c0
        c_ = []
        at = tf.nn.embedding_lookup
        for i in range(seq_len):
            c = at(f, i) * c + (1.0 - at(f, i)) * at(cx, i)
            c_.append(c)
        c_ = tf.stack(c_)

        h = r * tf.tanh(c_) + (1.0 - r) * hx

        h = tf.transpose(h, perm=[1, 0, 2])
        return h

    return unit, block

def attention(params, batch_size, feature_dim0, feature_dim1):
    init_matrix = xavier_initializer()
    Wp = tf.Variable(init_matrix([feature_dim0, feature_dim0]))
    W = tf.Variable(init_matrix([feature_dim1, feature_dim0]))
    b = tf.Variable(tf.constant(0.0, shape=[feature_dim0]))
    Wa = tf.Variable(init_matrix([feature_dim0, 1]))
    params.extend([Wp, W, b, Wa])

    def unit(bag_of_feature, current):
        current = tf.matmul(current, W) + b
        current = tf.reshape(current, [batch_size, 1, -1])
        bag_of_feature = tf.reshape(tf.matmul(tf.reshape(bag_of_feature, [-1, feature_dim0]), Wp), [batch_size, -1, feature_dim0])
        hidden = tf.nn.relu(bag_of_feature + current)

        alpha_logit = tf.reshape(tf.matmul(tf.reshape(hidden, [-1, feature_dim0]), Wa), [batch_size, -1])

        alpha = tf.nn.softmax(alpha_logit)

        reduced_feature = tf.reshape(tf.matmul(tf.reshape(alpha, [batch_size, 1, -1]), bag_of_feature), [batch_size, feature_dim0])

        return reduced_feature, alpha

    return unit

def get_policy(params, hidden_dim, context_dim, vocab_size, goal_size):
    init_matrix = xavier_initializer()
    W = tf.Variable(init_matrix([hidden_dim, vocab_size*goal_size]))
    U = tf.Variable(init_matrix([hidden_dim, hidden_dim]))
    Wc = tf.Variable(init_matrix([context_dim, hidden_dim]))
    b = tf.Variable(tf.constant(0.0, shape=[vocab_size*goal_size]))
    params.extend([W, Wc, b])

    def unit(hidden_state, context, goal):
        sub_feature = tf.matmul(hidden_state + tf.matmul(context, Wc), W)
        sub_feature = tf.reshape(sub_feature, [-1, vocab_size, goal_size])
        goal = tf.reshape(goal, [-1, 1, goal_size])
        logit = tf.reduce_sum(sub_feature * goal, -1)
        policy = tf.nn.softmax(logit)
        return policy, logit

    return unit

def MLP(params, dim0, dim1):
    init_matrix = xavier_initializer()
    W = tf.Variable(init_matrix([dim0, dim1]))
    b = tf.Variable(tf.constant(0.0, shape=[dim1]))
    params.extend([W, b])

    def unit(x):
        return tf.matmul(x, W) + b

    return unit

def goal_output(params, hidden_dim, img_feat, goal_size):
    init_matrix = xavier_initializer()
    Wl = tf.Variable(init_matrix([hidden_dim, goal_size]))
    Wi = tf.Variable(init_matrix([img_feat, goal_size]))
    b = tf.Variable(tf.constant(0.0, shape=[goal_size]))
    params.extend([Wl, Wi, b])

    def unit(x, im):
        return tf.matmul(x, Wl) + tf.matmul(im, Wi) + b

    return unit

class HierarchicalLanguageModel(object):
    def __init__(self, batch_size, hidden_dim, ResNet50_ref, goal_size, sequence_length, vocab_size):
        init_matrix = xavier_initializer()
        self.d_params = []
        self.g_params = []
        self.ResNet = ResNet50_ref

        self.batch_size = batch_size
        self.d_layer = 1
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        self.image_feature = ResNet50_ref.conv_feats
        self.image_feature_ave = tf.reduce_mean(self.image_feature, axis=1)
        self.image_feature_dim = ResNet50_ref.conv_feat_shape[1]
        self.image_feature_num = ResNet50_ref.conv_feat_shape[0]

        self.fake_x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
        self.dropout_keeprate = tf.placeholder(tf.float32)
        fake_x = tf.transpose(self.fake_x, perm=[1, 0])
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
        x = tf.transpose(self.x, perm=[1, 0])
        self.word_emb = tf.Variable(initial_value=init_matrix([vocab_size, hidden_dim]))
        self.g_params.append(self.word_emb)

        self.g_recur, _ = SRUCell(self.g_params, hidden_dim, self.image_feature_dim + hidden_dim)
        self.g_output = get_policy(self.g_params, hidden_dim, self.image_feature_dim, vocab_size, goal_size)
        self.g_attention = attention(self.g_params, batch_size, self.image_feature_dim, hidden_dim)
        self.d_recur_block = [None for _ in range(self.d_layer + 1)]
        self.d_recur_cell = [None for _ in range(self.d_layer + 1)]
        for i in range(self.d_layer):
            self.d_recur_cell[i], self.d_recur_block[i] = SRUCell(self.d_params, hidden_dim, hidden_dim)
        for i in range(self.d_layer, self.d_layer + 1):
            self.d_recur_cell[i], self.d_recur_block[i] = SRUCell(self.g_params, hidden_dim, hidden_dim)

        self.get_goal = goal_output(self.g_params, hidden_dim, self.image_feature_dim, goal_size)
        self.classify = MLP(self.d_params, hidden_dim, 1)
        at = tf.nn.embedding_lookup

        c0 = tf.zeros([batch_size, hidden_dim])
        start_token = tf.zeros([batch_size], dtype=tf.int32)
        c = c0

        last_token = start_token
        input = tf.concat([self.image_feature_ave, at(self.word_emb, start_token)], -1)
        print(input.shape)
        c_d = [c0] * (self.d_layer + 1)
        h_d = [None] * (self.d_layer + 1)
        policys = []
        logits = []
        goals = []
        alphas = []
        # Get prediction
        for i in range(sequence_length):
            c, h = self.g_recur(input, c)
            d_input = at(self.word_emb, last_token)
            for f in range(self.d_layer + 1):
                c_d[f], h_d[f] = self.d_recur_cell[f](d_input, c_d[f])
                d_input = h_d[f]

            context, alpha = self.g_attention(self.image_feature, h)
            # layer 3 for classification, layer 4 for guidance
            goal = tf.nn.l2_normalize(self.get_goal(d_input, context), dim=-1)
            goals.append(goal)
            alphas.append(alpha)
            policy, logit = self.g_output(h, context, goal)
            policys.append(policy)
            logits.append(logit)
            last_token = at(x, i)
            input = tf.concat([context, at(self.word_emb, last_token)], -1)
        logits = tf.stack(logits)
        self.alphas = tf.stack(alphas)
        alpha_loss = tf.reduce_mean(tf.reduce_sum((tf.reduce_sum(self.alphas, axis=0) - 1.0) ** 2, axis=-1))
        self.goals = tf.stack(goals)
        self.pretrain_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=x)) + 0.01 * alpha_loss
        self.pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pretrain_loss, self.g_params), 5.0)
        self.pretrain_op = tf.train.AdamOptimizer(1e-3)
        self.pretrain_update = self.pretrain_op.apply_gradients(zip(self.pretrain_grad, self.g_params))

        # Do classification
        last_token = start_token
        c_d = [c0] * self.d_layer
        h_d = [None] * self.d_layer
        cross_entropys = []
        seq = [x, fake_x]
        for is_fake in range(2):
            for i in range(sequence_length):
                d_input = at(self.word_emb, last_token)
                for f in range(self.d_layer):
                    c_d[f], h_d[f] = self.d_recur_cell[f](d_input, c_d[f])
                    d_input = h_d[f]
                last_token = at(seq[is_fake], i)
            d_input = tf.nn.dropout(d_input, 0.5)
            logit = self.classify(d_input)
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims([(1.0 - is_fake)]*batch_size, -1), logits=logit)
            cross_entropys.append(cross_entropy)
        self.d_loss = tf.reduce_mean(cross_entropys)
        self.d_op = tf.train.AdamOptimizer(1e-4)
        self.d_update = self.d_op.minimize(var_list=self.d_params, loss=self.d_loss)

        gen_x = []
        c = c0
        last_token = start_token
        input = tf.concat([self.image_feature_ave, at(self.word_emb, start_token)], -1)

        c_d = [c0] * (self.d_layer + 1)
        h_d = [None] * (self.d_layer + 1)
        # Do generation
        for i in range(sequence_length):
            c, h = self.g_recur(input, c)
            d_input = at(self.word_emb, last_token)
            for f in range(self.d_layer + 1):
                c_d[f], h_d[f] = self.d_recur_cell[f](d_input, c_d[f])
                d_input = h_d[f]
            # layer 3 for classification, layer 4 for guidance
            context, alpha = self.g_attention(self.image_feature, h)
            goal = tf.nn.l2_normalize(self.get_goal(h_d[self.d_layer], context), dim=-1)
            policy, logit = self.g_output(h, context, goal)
            last_token = tf.cast(tf.argmax(policy, axis=-1), tf.int32)
            gen_x.append(last_token)
            input = tf.concat([context, at(self.word_emb, last_token)], -1)
        self.gen_x = tf.transpose(tf.stack(gen_x), perm=[1, 0])

class HACap(object):
    def __init__(self, data_loader):
        self.d_params = []
        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size
        self.ResNet50 = ResNet50(self.batch_size)
        self.sequence_length = data_loader.max_len
        self.goal_size = 6
        self.step_size = 4
        self.rollout_num = 6
        self.f_params = self.ResNet50.params
        self.text_dropout = 0.75
        hidden_dim = 256

        self.model = HierarchicalLanguageModel(self.batch_size,
                                               hidden_dim=hidden_dim,
                                               ResNet50_ref=self.ResNet50,
                                               goal_size=self.goal_size,
                                               sequence_length=self.sequence_length,
                                               vocab_size=self.data_loader.vocab_size)

    def train_via_MLE(self, sess, image_caption):
        image, caption = image_caption

        feed_dict = {self.ResNet50.imgs: image, self.model.x: caption, K.learning_phase(): 0}

        _ = sess.run(self.model.pretrain_update, feed_dict)

    def train_discriminator(self, sess, image_caption):
        image, caption = image_caption

        feed_dict = {self.ResNet50.imgs: image, K.learning_phase(): 0}
        fake_caption = sess.run(self.model.gen_x, feed_dict)

        feed_dict = {self.model.x : caption, self.model.fake_x: fake_caption}
        _ = sess.run(self.model.d_update, feed_dict)

    def train_via_reinforce(self, sess, image_caption):
        pass

    def generate_caption(self, sess, image_caption, is_train=1.0, is_realized=True):
        image, _ = image_caption
        feed_dict = {self.ResNet50.imgs: image, K.learning_phase(): 0}
        caption = sess.run(self.model.gen_x, feed_dict)
        return self.ind_to_str(caption) if is_realized else caption

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
        feed_dict = {self.ResNet50.imgs: image, self.model.x: seq, K.learning_phase(): 0}
        alpha = sess.run(self.model.alphas, feed_dict)
        return alpha

    def get_guidance(self, sess, image_caption):
        image, seq = image_caption
        feed_dict = {self.ResNet50.imgs: image, self.model.x: seq, K.learning_phase(): 0}
        guidance = sess.run(self.model.goals, feed_dict)
        return guidance
