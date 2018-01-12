import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output

class Discriminator(object):
    def __init__(self, sequence_length, num_classes, vocab_size,dis_emb_dim,filter_sizes, num_filters,batch_size,hidden_dim, start_token,goal_out_size,step_size,l2_reg_lambda=0.0):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.dis_emb_dim = dis_emb_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.l2_reg_lambda = l2_reg_lambda
        self.num_filters_total = sum(self.num_filters)
        self.temperature = 1.0
        self.grad_clip = 5.0
        self.goal_out_size = goal_out_size
        self.step_size = step_size
        self.attention_size = 10

        self.D_input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.D_input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope('D_update'):
            self.D_l2_loss = tf.constant(0.0)
            self.FeatureExtractor_unit = self.FeatureExtractor()

            # Train for Discriminator
            with tf.variable_scope("feature") as self.feature_scope:
                D_feature = self.FeatureExtractor_unit(self.D_input_x,self.dropout_keep_prob)#,self.dropout_keep_prob)
                self.feature_scope.reuse_variables()
            # tf.get_variable_scope().reuse_variables()

            D_scores, D_predictions,self.ypred_for_auc = self.classification(D_feature)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=D_scores, labels=self.D_input_y)
            self.D_loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.D_l2_loss

            self.D_params = [param for param in tf.trainable_variables() if
                             'Discriminator' or 'FeatureExtractor' in param.name]
            d_optimizer = tf.train.AdamOptimizer(5e-5)
            D_grads_and_vars = d_optimizer.compute_gradients(self.D_loss, self.D_params, aggregation_method=2)
            self.D_train_op = d_optimizer.apply_gradients(D_grads_and_vars)


    # This module used to Extract sentence's Feature
    def FeatureExtractor(self):
        # Embedding layer
        # scope.reuse_variables()
        def unit(Feature_input,dropout_keep_prob):#,dropout_keep_prob):
            with tf.variable_scope('FeatureExtractor') as scope:
                with tf.device('/cpu:0'), tf.name_scope("embedding") as scope:
                    #
                    W_fe = tf.get_variable(
                        name="W_fe",
                        initializer=tf.random_uniform([self.vocab_size + 1, self.dis_emb_dim], -1.0, 1.0))
                    # scope.reuse_variables()
                    embedded_chars = tf.nn.embedding_lookup(W_fe, Feature_input + 1)
                    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

                # Create a convolution + maxpool layer for each filter size
                pooled_outputs = []
                for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
                    with tf.name_scope("conv-maxpool-%s" % filter_size) as scope:
                        # Convolution Layer
                        filter_shape = [filter_size, self.dis_emb_dim, 1, num_filter]
                        W = tf.get_variable(name="W-%s" % filter_size,
                                            initializer=tf.truncated_normal(filter_shape, stddev=0.1))
                        b = tf.get_variable(name="b-%s" % filter_size,
                                            initializer=tf.constant(0.1, shape=[num_filter]))
                        # scope.reuse_variables()
                        conv = tf.nn.conv2d(
                            embedded_chars_expanded,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv-%s" % filter_size)
                        # Apply nonlinearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-%s" % filter_size)
                        # Maxpooling over the outputs
                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool-%s" % filter_size)
                        pooled_outputs.append(pooled)
                        #
                # Combine all the pooled features
                h_pool = tf.concat(pooled_outputs, 3)
                h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])

                # Add highway
                with tf.name_scope("highway"):
                    h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0)

                    # Add dropout
                with tf.name_scope("dropout"):
                    h_drop = tf.nn.dropout(h_highway,dropout_keep_prob)

            return h_drop

        return unit

    def classification(self, D_input):
        with tf.variable_scope('Discriminator'):
            W_d = tf.Variable(tf.truncated_normal([self.num_filters_total, self.num_classes], stddev=0.1), name="W")
            b_d = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.D_l2_loss += tf.nn.l2_loss(W_d)
            self.D_l2_loss += tf.nn.l2_loss(b_d)
            self.scores = tf.nn.xw_plus_b(D_input, W_d, b_d, name="scores")
            self.ypred_for_auc = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        return self.scores, self.predictions, self.ypred_for_auc


class LeakGAN(object):
    def __init__(self, sequence_length, num_classes, vocab_size,
            emb_dim, dis_emb_dim, filter_sizes, num_filters, batch_size, hidden_dim, start_token, goal_out_size,
                 goal_size, step_size, D_model, ResNet_model, LSTMlayer_num=1, l2_reg_lambda=0.0, learning_rate=0.01):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.dis_emb_dim = dis_emb_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.LSTMlayer_num = LSTMlayer_num
        self.l2_reg_lambda = l2_reg_lambda
        self.learning_rate = learning_rate
        self.num_filters_total = sum(self.num_filters)
        self.grad_clip = 5.0
        self.goal_out_size = goal_out_size
        self.goal_size = goal_size
        self.step_size = step_size
        self.D_model = D_model
        self.attention_size = 20
        self.ResNet_model = ResNet_model
        self.image_feature = ResNet_model.conv_feats
        self.image_feature_ave = tf.reduce_mean(self.image_feature, axis=1)
        self.image_feature_dim = ResNet_model.conv_feat_shape[1]
        self.image_feature_num = ResNet_model.conv_feat_shape[0]
        self.FeatureExtractor_unit = self.D_model.FeatureExtractor_unit

        self.scope = self.D_model.feature_scope
        self.worker_params = [] # + ResNet_model.params
        self.manager_params = [] # + ResNet_model.params

        self.epis = 0.65
        self.tem = 10.0
        with tf.variable_scope('place_holder'):
            self.x = tf.placeholder(tf.int32, shape=[self.batch_size,self.sequence_length])  # sequence of tokens generated by generator
            self.reward = tf.placeholder(tf.float32, shape=[self.batch_size,self.sequence_length / self.step_size])  # sequence of tokens generated by generator
            self.given_num = tf.placeholder(tf.int32)
            self.drop_out = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.train = tf.placeholder(tf.int32, None, name="train")

        with tf.variable_scope('Worker'):
            self.g_embeddings = tf.Variable(self.build_mat([self.vocab_size, self.emb_dim], stddev=0.1))
            self.worker_params.append(self.g_embeddings)
            self.g_worker_recurrent_unit = self.create_Worker_recurrent_unit(self.worker_params)  # maps h_tm1 to h_t for generator
            self.g_worker_output_unit = self.create_Worker_output_unit(self.worker_params)  # maps h_t to o_t (output token logits)
            self.W_workerOut_change = tf.Variable(self.build_mat([self.vocab_size, self.goal_size], stddev=0.1))

            self.g_change = tf.Variable(self.build_mat([self.goal_out_size, self.goal_size], stddev=0.1))
            self.worker_params.extend([self.W_workerOut_change,self.g_change])
            self.h0_manager = tf.zeros([2, self.batch_size, self.hidden_dim])
            self.dropout_output = self.create_dropout_unit(self.worker_params)

        with tf.variable_scope('Manager'):
            self.g_manager_recurrent_unit = self.create_Manager_recurrent_unit(self.manager_params)  # maps h_tm1 to h_t for generator
            self.g_manager_output_unit = self.create_Manager_output_unit(self.manager_params)  # maps h_t to o_t (output token logits)
            self.h0_worker = tf.zeros([2, self.batch_size, self.hidden_dim])
            self.goal_init = tf.get_variable("goal_init",initializer=tf.truncated_normal([self.batch_size,self.goal_out_size], stddev=0.1))
            self.manager_params.extend([self.goal_init])

        self.padding_array = tf.constant(-1, shape=[self.batch_size, self.sequence_length], dtype=tf.int32)
        with tf.name_scope("attention"):
            self.attention = self.create_attention_unit([self.worker_params])
        with tf.name_scope("roll_out"):
            self.gen_for_reward = self.rollout(self.x,self.given_num)

        # processed for batch
        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x),perm=[1, 0, 2])  # seq_length x batch_size x emb_dim

        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                                 dynamic_size=False, infer_shape=True)
        alphas = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32,size=1,dynamic_size=True, infer_shape=True,clear_after_read = False)

        goal = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                                 dynamic_size=False, infer_shape=True,clear_after_read = False)

        feature_array = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length+1,
                                                     dynamic_size=False, infer_shape=True, clear_after_read=False)
        real_goal_array = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length/self.step_size,
                                                       dynamic_size=False, infer_shape=True, clear_after_read=False)

        gen_real_goal_array = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                                       dynamic_size=False, infer_shape=True, clear_after_read=False)

        gen_o_worker_array = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length/self.step_size,
                                                       dynamic_size=False, infer_shape=True, clear_after_read=False)

        def _g_recurrence(i, x_t,h_tm1,h_tm1_manager, gen_o, gen_x,goal,last_goal,real_goal,step_size,gen_real_goal_array,gen_o_worker_array):
            ## padding sentence by -1
            cur_sen = tf.cond(i > 0,lambda:tf.split(tf.concat([tf.transpose(gen_x.stack(), perm=[1, 0]),self.padding_array],1),[self.sequence_length,i],1)[0],lambda :self.padding_array)
            with tf.variable_scope(self.scope):
                feature = self.FeatureExtractor_unit(cur_sen,self.drop_out)
            alpha = self.attention(self.image_feature, tf.nn.embedding_lookup(h_tm1, 0))
            focus = tf.reshape(
                tf.matmul(tf.reshape(alpha, [self.batch_size, 1, self.image_feature_num]),
                          self.image_feature), [self.batch_size, self.image_feature_dim])
            h_t_Worker = self.g_worker_recurrent_unit(x_t, h_tm1, focus)  # hidden_memory_tuple
            o_t_Worker = self.g_worker_output_unit(h_t_Worker, focus, x_t)  # batch x vocab , logits not prob
            o_t_Worker = tf.reshape(o_t_Worker,[self.batch_size,self.vocab_size,self.goal_size])

            h_t_manager = self.g_manager_recurrent_unit(feature, h_tm1_manager, focus)
            sub_goal = self.g_manager_output_unit(h_t_manager, focus, x_t)
            sub_goal = tf.nn.l2_normalize(sub_goal, 1)
            goal = goal.write(i,sub_goal)

            real_sub_goal = tf.add(last_goal,sub_goal)

            w_g = tf.matmul(real_goal,self.g_change)   #batch x goal_size
            w_g = tf.nn.l2_normalize(w_g, 1)
            gen_real_goal_array = gen_real_goal_array.write(i,real_goal)

            w_g = tf.expand_dims(w_g,2)  #batch x goal_size x 1

            gen_o_worker_array = gen_o_worker_array.write(i,o_t_Worker)

            x_logits = tf.matmul(o_t_Worker,w_g)
            # x_logits = tf.norm(o_t_Worker, ord=1, axis=-1)
            x_logits = tf.squeeze(x_logits)
            x_logits = self.dropout_output(x_logits, focus, x_t)

            log_prob = tf.log(tf.nn.softmax(
                tf.cond(i > 1, lambda: tf.cond(self.train > 0, lambda: self.tem, lambda: 1.0), lambda: 1.0) * x_logits))

            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            with tf.control_dependencies([cur_sen]):
                gen_x = gen_x.write(i, next_token)  # indices, batch_size
            gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.vocab_size, 1.0, 0.0),
                                                             tf.nn.softmax(x_logits)), 1))  # [batch_size] , prob
            return i+1,x_tp1,h_t_Worker,h_t_manager,gen_o,gen_x,goal,\
                   tf.cond(((i+1)%step_size)>0,lambda:real_sub_goal,lambda :tf.constant(0.0,shape=[self.batch_size,self.goal_out_size]))\
                    ,tf.cond(((i+1)%step_size)>0,lambda :real_goal,lambda :real_sub_goal),step_size,gen_real_goal_array,gen_o_worker_array

        _, _, _,_, self.gen_o, self.gen_x,_,_,_,_,self.gen_real_goal_array,self.gen_o_worker_array= control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4,_5,_6,_7,_8,_9,_10,_11: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),self.h0_worker,self.h0_manager,
                        gen_o, gen_x,goal,tf.zeros([self.batch_size,self.goal_out_size]),self.goal_init,step_size,gen_real_goal_array,gen_o_worker_array),parallel_iterations=1)

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size

        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length

        self.gen_real_goal_array = self.gen_real_goal_array.stack()  # seq_length x batch_size x goal

        self.gen_real_goal_array = tf.transpose(self.gen_real_goal_array, perm=[1, 0,2])  # batch_size x seq_length x goal

        self.gen_o_worker_array = self.gen_o_worker_array.stack()  # seq_length x batch_size* vocab*goal

        self.gen_o_worker_array = tf.transpose(self.gen_o_worker_array, perm=[1, 0,2,3])  # batch_size x seq_length * vocab*goal

        sub_feature = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length/self.step_size,
                                                       dynamic_size=False, infer_shape=True, clear_after_read=False)

        all_sub_features = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                                   dynamic_size=False, infer_shape=True, clear_after_read=False)
        all_sub_goals = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                                   dynamic_size=False, infer_shape=True, clear_after_read=False)

        # supervised pretraining for generator
        g_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)
        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)


        def preTrain(i, alphas, x_t,g_predictions,h_tm1,input_x,h_tm1_manager,last_goal,real_goal,feature_array,real_goal_array,sub_feature,all_sub_features,all_sub_goals):
            ## padding sentence by -1
            cur_sen = tf.split(tf.concat([tf.split(input_x,[i,self.sequence_length-i],1)[0],self.padding_array],1),[self.sequence_length,i],1)[0]  #padding sentence
            with tf.variable_scope(self.scope):
                feature = self.FeatureExtractor_unit(cur_sen,self.drop_out)
            alpha = self.attention(self.image_feature, tf.nn.embedding_lookup(h_tm1, 0))
            alphas = tf.cond(i > 0, lambda: alphas.write(i - 1, alpha), lambda: alphas)
            focus = tf.reshape(
                tf.matmul(tf.reshape(alpha, [self.batch_size, 1, self.image_feature_num]),
                          self.image_feature), [self.batch_size, self.image_feature_dim])
            feature_array = feature_array.write(i,feature)

            real_goal_array = tf.cond(i>0, lambda: real_goal_array,
                                       lambda: real_goal_array.write(0, self.goal_init))
            h_t_manager = self.g_manager_recurrent_unit(feature, h_tm1_manager, focus)
            sub_goal = self.g_manager_output_unit(h_t_manager, focus, x_t)
            sub_goal = tf.nn.l2_normalize(sub_goal, 1)

            h_t_Worker = tf.cond(i>0,lambda :self.g_worker_recurrent_unit(x_t, h_tm1, focus),
                                     lambda : h_tm1)# hidden_memory_tuple
            o_t_Worker = self.g_worker_output_unit(h_t_Worker, focus, x_t)  # batch x vocab , logits not prob
            o_t_Worker = tf.reshape(o_t_Worker, [self.batch_size, self.vocab_size, self.goal_size])

            real_sub_goal =tf.cond(i>0,lambda :tf.add(last_goal, sub_goal),
                                       lambda :real_goal)
            all_sub_goals = tf.cond(i > 0,lambda: all_sub_goals.write(i-1, real_goal),
                                        lambda: all_sub_goals)

            w_g = tf.matmul(real_goal, self.g_change)  # batch x goal_size
            w_g = tf.nn.l2_normalize(w_g, 1)
            w_g = tf.expand_dims(w_g, 2)  # batch x goal_size x 1

            x_logits = tf.matmul(o_t_Worker, w_g)
            # x_logits = tf.norm(o_t_Worker, ord=1, axis=-1)
            x_logits = tf.squeeze(x_logits)
            x_logits = self.dropout_output(x_logits, focus, x_t)

            g_predictions = tf.cond(i>0,lambda :g_predictions.write(i-1, tf.nn.softmax(x_logits)),lambda :g_predictions)

            sub_feature = tf.cond(((((i) % step_size) > 0)),
                               lambda: sub_feature,
                               lambda: (tf.cond(i > 0, lambda:sub_feature.write(i/step_size-1,tf.subtract(feature, feature_array.read(i - step_size))),
                                                       lambda: sub_feature)))

            all_sub_features = tf.cond(i > 0,lambda: tf.cond((i % step_size) > 0, lambda :all_sub_features.write(i-1,tf.subtract(feature,feature_array.read(i-i%step_size))),\
                                                                                     lambda :all_sub_features.write(i-1,tf.subtract(feature,feature_array.read(i-step_size)))),
                                            lambda : all_sub_features)

            real_goal_array = tf.cond(((i) % step_size) > 0, lambda: real_goal_array,
                                                            lambda: tf.cond((i)/step_size  < self.sequence_length/step_size,
                                                                        lambda :tf.cond(i>0,lambda :real_goal_array.write((i)/step_size, real_sub_goal),
                                                                                            lambda :real_goal_array),
                                                                        lambda :real_goal_array))
            x_tp1 = tf.cond(i>0,lambda :ta_emb_x.read(i-1),
                                lambda :x_t)

            return i+1, alphas, x_tp1, g_predictions, h_t_Worker, input_x, h_t_manager,\
                   tf.cond(((i)%step_size)>0,lambda:real_sub_goal,lambda :tf.constant(0.0,shape=[self.batch_size,self.goal_out_size])) ,\
                    tf.cond(((i) % step_size) > 0, lambda: real_goal, lambda: real_sub_goal),\
                   feature_array,real_goal_array,sub_feature,all_sub_features,all_sub_goals

        _, self.alphas, _, self.g_predictions, _,_,_,_,_, self.feature_array, self.real_goal_array,self.sub_feature,self.all_sub_features,self.all_sub_goals = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6,_7,_8,_9,_10,_11,_12, _13: i < self.sequence_length+1,
            body=preTrain,
            loop_vars=(tf.constant(0, dtype=tf.int32), alphas, tf.nn.embedding_lookup(self.g_embeddings, self.start_token),g_predictions,self.h0_worker,
                      self.x, self.h0_manager, tf.zeros([self.batch_size, self.goal_out_size]),self.goal_init, feature_array,real_goal_array,sub_feature,all_sub_features,all_sub_goals),
            parallel_iterations=1)

        self.alphas = self.alphas.stack() # [batch_size x sequence_length x num_feat]
        alpha_loss = tf.reduce_mean(tf.reduce_sum((tf.reduce_sum(self.alphas, axis=1) - 1.0)**2, axis=-1))
        self.sub_feature = self.sub_feature.stack() # seq_length x batch_size x num_filter
        self.sub_feature = tf.transpose(self.sub_feature, perm=[1, 0, 2])

        self.real_goal_array = self.real_goal_array.stack()
        self.real_goal_array = tf.transpose(self.real_goal_array, perm=[1, 0, 2])
        self.pretrain_goal_loss = -tf.reduce_sum(1-tf.losses.cosine_distance(tf.nn.l2_normalize(self.sub_feature,2),tf.nn.l2_normalize(self.real_goal_array,2),2)
        ) / (self.sequence_length * self.batch_size/self.step_size)

        with tf.name_scope("Manager_PreTrain_update"):
            pretrain_manager_opt = tf.train.AdamOptimizer(self.learning_rate)

            self.pretrain_manager_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pretrain_goal_loss, self.manager_params), self.grad_clip)
            self.pretrain_manager_updates = pretrain_manager_opt.apply_gradients(zip(self.pretrain_manager_grad, self.manager_params))
        # self.real_goal_array = self.real_goal_array.stack()

        self.g_predictions = tf.transpose(self.g_predictions.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size
        self.cross_entropy = tf.reduce_sum(self.g_predictions * tf.log(tf.clip_by_value(self.g_predictions, 1e-20, 1.0))) / (
        self.batch_size * self.sequence_length * self.vocab_size)

        self.pretrain_worker_loss = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.vocab_size, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.vocab_size]), 1e-20, 1.0)
            )
        ) / (self.sequence_length * self.batch_size) + alpha_loss * 0.00001

        with tf.name_scope("Worker_PreTrain_update"):
            # training updates
            pretrain_worker_opt = tf.train.AdamOptimizer(self.learning_rate)

            self.pretrain_worker_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pretrain_worker_loss, self.worker_params), self.grad_clip)
            self.pretrain_worker_updates = pretrain_worker_opt.apply_gradients(zip(self.pretrain_worker_grad, self.worker_params))

        self.goal_loss = -tf.reduce_sum(tf.multiply(self.reward,1-tf.losses.cosine_distance(tf.nn.l2_normalize(self.sub_feature,2), tf.nn.l2_normalize(self.real_goal_array,2), 2)
                                                 )) / (self.sequence_length * self.batch_size / self.step_size)

        with tf.name_scope("Manager_update"):
            manager_opt = tf.train.AdamOptimizer(self.learning_rate)

            self.manager_grad, _ = tf.clip_by_global_norm(
                tf.gradients(self.goal_loss, self.manager_params), self.grad_clip)
            self.manager_updates = manager_opt.apply_gradients(
                zip(self.manager_grad, self.manager_params))


        self.all_sub_features = self.all_sub_features.stack()
        self.all_sub_features = tf.transpose(self.all_sub_features, perm=[1, 0, 2])

        self.all_sub_goals = self.all_sub_goals.stack()
        self.all_sub_goals = tf.transpose(self.all_sub_goals, perm=[1, 0, 2])
        # self.all_sub_features = tf.nn.l2_normalize(self.all_sub_features, 2)
        self.Worker_Reward = 1-tf.losses.cosine_distance(tf.nn.l2_normalize(self.all_sub_features,2), tf.nn.l2_normalize(self.all_sub_goals,2), 2)
        # print self.Worker_Reward.shape
        self.worker_loss = -tf.reduce_sum(
            tf.multiply(self.Worker_Reward , tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.vocab_size, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.vocab_size]), 1e-20, 1.0))
            )
        ) / (self.sequence_length * self.batch_size)
        with tf.name_scope("Worker_update"):
            # training updates
            worker_opt = tf.train.AdamOptimizer(self.learning_rate)
            self.worker_grad, _ = tf.clip_by_global_norm(
                tf.gradients(self.worker_loss, self.worker_params), self.grad_clip)
            self.worker_updates = worker_opt.apply_gradients(
                zip(self.worker_grad, self.worker_params))

    def rollout(self,input_x,given_num):
        with tf.device("/cpu:0"):
            processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings,input_x),perm=[1, 0, 2])  # seq_length x batch_size x emb_dim
        ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(processed_x)

        #Next is rollout
        gen_for_reward = tensor_array_ops.TensorArray(dtype=tf.int32, size=1, dynamic_size=True, infer_shape=True,clear_after_read=False)
        ta_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length)
        ta_x = ta_x.unstack(tf.transpose(input_x, perm=[1, 0]))

        # When current index i < given_num, use the provided tokens as the input at each time step
        def _g_recurrence_1(i, x_t,input_x,gen_x,h_tm1,h_tm1_manager,last_goal,real_goal,give_num):

            cur_sen = tf.split(tf.concat([tf.split(input_x, [i, self.sequence_length - i], 1)[0], self.padding_array], 1),[self.sequence_length, i], 1)[0]
            with tf.variable_scope(self.scope):
                feature = self.FeatureExtractor_unit(cur_sen,self.drop_out)
            alpha = self.attention(self.image_feature, tf.nn.embedding_lookup(h_tm1, 0))
            focus = tf.reshape(
                tf.matmul(tf.reshape(alpha, [self.batch_size, 1, self.image_feature_num]),
                          self.image_feature), [self.batch_size, self.image_feature_dim])
            h_t_manager = self.g_manager_recurrent_unit(feature, h_tm1_manager, focus)
            sub_goal = self.g_manager_output_unit(h_t_manager, focus, x_t)
            sub_goal = tf.nn.l2_normalize(sub_goal, 1)

            h_t_Worker = tf.cond(i > 0, lambda: self.g_worker_recurrent_unit(x_t, h_tm1, focus),
                                 lambda: h_tm1)  # hidden_memory_tuple

            real_sub_goal = tf.cond(i > 0, lambda: tf.add(last_goal, sub_goal), lambda: real_goal)
            # real_goal_array = real_goal_array.write(i, real_sub_goal)

            x_tp1 = tf.cond(i > 0, lambda: ta_emb_x.read(i - 1), lambda: x_t)

            # hidden_memory_tuple
            with tf.control_dependencies([cur_sen]):
                gen_x = tf.cond(i > 0, lambda :gen_x.write(i-1, ta_x.read(i-1)),lambda :gen_x)
            return i + 1, x_tp1,input_x,gen_x,h_t_Worker, h_t_manager, \
                   tf.cond(((i) % self.step_size) > 0, lambda: real_sub_goal,
                           lambda: tf.constant(0.0, shape=[self.batch_size, self.goal_out_size])), \
                   tf.cond(((i) % self.step_size) > 0, lambda: real_goal, lambda: real_sub_goal), give_num

        # When current index i >= given_num, start roll-out, use the output as time step t as the input at time step t+1
        def _g_recurrence_2(i, x_t,gen_x,h_tm1,h_tm1_manager,last_goal,real_goal):
            # with tf.device('/cpu:0'):
            cur_sen = tf.cond(i > 0,lambda:tf.split(tf.concat([tf.transpose(gen_x.stack(), perm=[1, 0]),self.padding_array],1),[self.sequence_length,i-1],1)[0],lambda :self.padding_array)
            with tf.variable_scope(self.scope):
                feature = self.FeatureExtractor_unit(cur_sen,self.drop_out)
            alpha = self.attention(self.image_feature, tf.nn.embedding_lookup(h_tm1, 0))
            focus = tf.reshape(
                tf.matmul(tf.reshape(alpha, [self.batch_size, 1, self.image_feature_num]),
                          self.image_feature), [self.batch_size, self.image_feature_dim])
            h_t_Worker = self.g_worker_recurrent_unit(x_t, h_tm1, focus)  # hidden_memory_tuple
            o_t_Worker = self.g_worker_output_unit(h_t_Worker, focus, x_t)  # batch x vocab , logits not prob

            o_t_Worker = tf.reshape(o_t_Worker, [self.batch_size, self.vocab_size, self.goal_size])
            # o_t_Worker = tf.expand_dims(o_t_Worker,2)   # batch x vocab x 1
            # o_t_Worker = tf.multiply(o_t_Worker,tf.nn.softmax(self.W_workerOut_change) ) #batch x vocab x goal_size

            h_t_manager = self.g_manager_recurrent_unit(feature, h_tm1_manager, focus)
            sub_goal = self.g_manager_output_unit(h_t_manager, focus, x_t)
            sub_goal = tf.nn.l2_normalize(sub_goal, 1)

            real_sub_goal = tf.add(last_goal,sub_goal)
            w_g = tf.matmul(real_goal,self.g_change)   #batch x goal_size
            w_g = tf.nn.l2_normalize(w_g, 1)
            w_g = tf.expand_dims(w_g,2)  #batch x goal_size x 1

            x_logits = tf.matmul(o_t_Worker, w_g)
            # x_logits = tf.norm(o_t_Worker, ord=1, axis=-1)
            x_logits = tf.squeeze(x_logits)
            x_logits = self.dropout_output(x_logits, focus, x_t)

            log_prob = tf.log(tf.nn.softmax(x_logits))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            with tf.control_dependencies([cur_sen]):
                gen_x = gen_x.write(i-1, next_token)  # indices, batch_size
            return i + 1, x_tp1, gen_x,h_t_Worker,h_t_manager,\
                    tf.cond(((i) % self.step_size) > 0, lambda: real_sub_goal,
                                                lambda: tf.constant(0.0, shape=[self.batch_size, self.goal_out_size])), \
                    tf.cond(((i) % self.step_size) > 0, lambda: real_goal, lambda: real_sub_goal)

        i, x_t,_, gen_for_reward,h_worker, h_manager, self.last_goal_for_reward,self.real_goal_for_reward,given_num  = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3,_4,_5,_6, _7,given_num: i < given_num+1,
            body=_g_recurrence_1,
            loop_vars=(tf.constant(0, dtype=tf.int32),tf.nn.embedding_lookup(self.g_embeddings, self.start_token),self.x,gen_for_reward,
                       self.h0_worker,self.h0_manager,tf.zeros([self.batch_size, self.goal_out_size]),self.goal_init,given_num),parallel_iterations=1)  ##input groud-truth

        _, _, gen_for_reward,_, _,_,_  = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4,_5,_6: i < self.sequence_length+1,
            body=_g_recurrence_2,
            loop_vars=(i, x_t, gen_for_reward,h_worker, h_manager,self.last_goal_for_reward,self.real_goal_for_reward),parallel_iterations=1)   ## rollout by original policy

        gen_for_reward = gen_for_reward.stack()  # seq_length x batch_size

        gen_for_reward = tf.transpose(gen_for_reward, perm=[1, 0])  # batch_size x seq_length

        return gen_for_reward


    def update_feature_function(self,D_model):
        self.FeatureExtractor_unit = D_model.FeatureExtractor_unit

    def create_attention_unit(self, params):
        self.Iatt = tf.Variable(self.build_mat([self.image_feature_dim, self.image_feature_dim]))
        self.Latt = tf.Variable(self.build_mat([self.hidden_dim, self.image_feature_dim]))
        self.Uatt = tf.Variable(self.build_mat([self.image_feature_dim, 1]))
        self.batt = tf.Variable(self.build_mat([self.image_feature_dim]))
        self.batt0 = tf.Variable(self.build_mat([1]))
        for param in params:
            param.extend([self.Uatt, self.Iatt, self.Latt, self.batt, self.batt0])
        def unit(image_feature, leaked_inf):
            feature = tf.reshape(tf.tanh(tf.matmul(
                tf.reshape(image_feature, [-1, self.image_feature_dim]), self.Iatt)),
                [self.batch_size, self.image_feature_num, -1]) + tf.reshape(tf.matmul(leaked_inf, self.Latt),
                                                    [self.batch_size, 1, -1]) + self.batt
            alpha = tf.reshape(
                tf.matmul(tf.reshape(feature, [-1, self.image_feature_dim]), self.Uatt) + self.batt0,
                [self.batch_size, self.image_feature_num])
            alpha = tf.nn.softmax(alpha)
            return alpha
        return unit
    def build_mat(self, shape, stddev=0.1):
        p = xavier_initializer()
        return p(shape)
    def create_Worker_recurrent_unit(self, params):
        with tf.variable_scope('Worker'):
            # Weights and Bias for input and hidden tensor
            self.Wi_worker = tf.Variable(self.build_mat([self.emb_dim, self.hidden_dim], stddev=0.1))
            self.Vi_worker = tf.Variable(self.build_mat([self.image_feature_dim, self.hidden_dim], stddev=0.1))
            self.Ui_worker = tf.Variable(self.build_mat([self.hidden_dim, self.hidden_dim], stddev=0.1))
            self.bi_worker = tf.Variable(self.build_mat([self.hidden_dim], stddev=0.1))

            self.Wf_worker = tf.Variable(self.build_mat([self.emb_dim, self.hidden_dim], stddev=0.1))
            self.Vf_worker = tf.Variable(self.build_mat([self.image_feature_dim, self.hidden_dim], stddev=0.1))
            self.Uf_worker = tf.Variable(self.build_mat([self.hidden_dim, self.hidden_dim], stddev=0.1))
            self.bf_worker = tf.Variable(self.build_mat([self.hidden_dim], stddev=0.1))

            self.Wog_worker = tf.Variable(self.build_mat([self.emb_dim, self.hidden_dim], stddev=0.1))
            self.Vog_worker = tf.Variable(self.build_mat([self.image_feature_dim, self.hidden_dim], stddev=0.1))
            self.Uog_worker = tf.Variable(self.build_mat([self.hidden_dim, self.hidden_dim], stddev=0.1))
            self.bog_worker = tf.Variable(self.build_mat([self.hidden_dim], stddev=0.1))

            self.Wc_worker = tf.Variable(self.build_mat([self.emb_dim, self.hidden_dim], stddev=0.1))
            self.Vc_worker = tf.Variable(self.build_mat([self.image_feature_dim, self.hidden_dim], stddev=0.1))
            self.Uc_worker = tf.Variable(self.build_mat([self.hidden_dim, self.hidden_dim], stddev=0.1))
            self.bc_worker = tf.Variable(self.build_mat([self.hidden_dim], stddev=0.1))
            params.extend([
                self.Wi_worker, self.Ui_worker, self.Vi_worker, self.bi_worker,
                self.Wf_worker, self.Uf_worker, self.Vf_worker, self.bf_worker,
                self.Wog_worker, self.Uog_worker, self.Vog_worker, self.bog_worker,
                self.Wc_worker, self.Uc_worker, self.Vc_worker, self.bc_worker])

            def unit(x, hidden_memory_tm1, focus):
                previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

                # Input Gate
                i = tf.sigmoid(
                    tf.matmul(x, self.Wi_worker) + tf.matmul(focus, self.Vi_worker) +
                    tf.matmul(previous_hidden_state, self.Ui_worker) + self.bi_worker
                )

                # Forget Gate
                f = tf.sigmoid(
                    tf.matmul(x, self.Wf_worker) + tf.matmul(focus, self.Vf_worker) +
                    tf.matmul(previous_hidden_state, self.Uf_worker) + self.bf_worker
                )

                # Output Gate
                o = tf.sigmoid(
                    tf.matmul(x, self.Wog_worker) + tf.matmul(focus, self.Vog_worker) +
                    tf.matmul(previous_hidden_state, self.Uog_worker) + self.bog_worker
                )

                # New Memory Cell
                c_ = tf.nn.tanh(
                    tf.matmul(x, self.Wc_worker) + tf.matmul(focus, self.Vc_worker) +
                    tf.matmul(previous_hidden_state, self.Uc_worker) + self.bc_worker
                )

                # Final Memory cell
                c = f * c_prev + i * c_

                # Current Hidden state
                current_hidden_state = o * tf.nn.tanh(c)

                return tf.stack([current_hidden_state, c])

            return unit

    def create_Worker_output_unit(self, params):
        with tf.variable_scope('Worker'):
            self.W_worker = tf.Variable(self.build_mat([self.hidden_dim, self.vocab_size*self.goal_size], stddev=0.1))
            # self.U_worker = tf.Variable(self.build_mat([self.image_feature_dim, self.vocab_size], stddev=0.1))
            self.b_worker = tf.Variable(self.build_mat([self.vocab_size*self.goal_size], stddev=0.1))
            params.extend([self.W_worker, self.b_worker])
            # params.extend([self.U_worker])
            self.WHigh = tf.Variable(self.build_mat([self.image_feature_dim, self.hidden_dim]))
            self.UHigh = tf.Variable(self.build_mat([self.emb_dim, self.hidden_dim]))
            params.extend([self.WHigh, self.UHigh])

            def unit(hidden_memory_tuple, focus, prev_x):
                hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
                # hidden_state : batch x hidden_dim
                hidden_state += tf.matmul(focus, self.WHigh) + tf.matmul(prev_x, self.UHigh)
                hidden_state = tf.nn.tanh(hidden_state)
                logits = tf.matmul(hidden_state, self.W_worker) + self.b_worker
                logits = tf.reshape(logits, [self.batch_size, self.vocab_size, self.goal_size])# * tf.expand_dims(tf.matmul(feature, self.U_worker), -1)
                # output = tf.nn.softmax(logits)
                return tf.reshape(logits, [self.batch_size, self.vocab_size*self.goal_size])

            return unit
    def create_dropout_unit(self, params):
        with tf.variable_scope('Worker'):
            def unit(base_logit, context, prev_x):
                return base_logit # + tf.matmul(context, self.WHigh) + tf.matmul(prev_x, self.UHigh)
        return unit
    def create_Manager_recurrent_unit(self, params):
        with tf.variable_scope('Manager'):
            # Weights and Bias for input and hidden tensor
            self.Wi = tf.Variable(self.build_mat([self.num_filters_total, self.hidden_dim], stddev=0.1))
            self.Vi = tf.Variable(self.build_mat([self.image_feature_dim, self.hidden_dim], stddev=0.1))
            self.Ui = tf.Variable(self.build_mat([self.hidden_dim, self.hidden_dim], stddev=0.1))
            self.bi = tf.Variable(self.build_mat([self.hidden_dim], stddev=0.1))

            self.Wf = tf.Variable(self.build_mat([self.num_filters_total, self.hidden_dim], stddev=0.1))
            self.Vf = tf.Variable(self.build_mat([self.image_feature_dim, self.hidden_dim], stddev=0.1))
            self.Uf = tf.Variable(self.build_mat([self.hidden_dim, self.hidden_dim], stddev=0.1))
            self.bf = tf.Variable(self.build_mat([self.hidden_dim], stddev=0.1))

            self.Wog = tf.Variable(self.build_mat([self.num_filters_total, self.hidden_dim], stddev=0.1))
            self.Vog = tf.Variable(self.build_mat([self.image_feature_dim, self.hidden_dim], stddev=0.1))
            self.Uog = tf.Variable(self.build_mat([self.hidden_dim, self.hidden_dim], stddev=0.1))
            self.bog = tf.Variable(self.build_mat([self.hidden_dim], stddev=0.1))

            self.Wc = tf.Variable(self.build_mat([self.num_filters_total, self.hidden_dim], stddev=0.1))
            self.Vc = tf.Variable(self.build_mat([self.image_feature_dim, self.hidden_dim], stddev=0.1))
            self.Uc = tf.Variable(self.build_mat([self.hidden_dim, self.hidden_dim], stddev=0.1))
            self.bc = tf.Variable(self.build_mat([self.hidden_dim], stddev=0.1))
            params.extend([
                self.Wi, self.Ui, self.bi,
                self.Wf, self.Uf, self.bf,
                self.Wog, self.Uog, self.bog,
                self.Wc, self.Uc, self.bc])
            params.extend([
                self.Vi, self.Vf, self.Vog, self.Vc
            ])

            def unit(x, hidden_memory_tm1, focus):
                previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

                # Input Gate
                i = tf.sigmoid(
                    tf.matmul(x, self.Wi) +  tf.matmul(focus, self.Vi) +
                    tf.matmul(previous_hidden_state, self.Ui) + self.bi
                )

                # Forget Gate
                f = tf.sigmoid(
                    tf.matmul(x, self.Wf) +  tf.matmul(focus, self.Vf) +
                    tf.matmul(previous_hidden_state, self.Uf) + self.bf
                )

                # Output Gate
                o = tf.sigmoid(
                    tf.matmul(x, self.Wog) + tf.matmul(focus, self.Vf) +
                    tf.matmul(previous_hidden_state, self.Uog) + self.bog
                )

                # New Memory Cell
                c_ = tf.nn.tanh(
                    tf.matmul(x, self.Wc) + tf.matmul(focus, self.Vi) +
                    tf.matmul(previous_hidden_state, self.Uc) + self.bc
                )

                # Final Memory cell
                c = f * c_prev + i * c_

                # Current Hidden state
                current_hidden_state = o * tf.nn.tanh(c)

                return tf.stack([current_hidden_state, c])

            return unit

    def create_Manager_output_unit(self, params):
        with tf.variable_scope('Manager'):
            self.Wo = tf.Variable(self.build_mat([self.hidden_dim, self.goal_out_size], stddev=0.1))
            self.bo = tf.Variable(self.build_mat([self.goal_out_size], stddev=0.1))
            params.extend([self.Wo, self.bo])
            self.WHigh_manager = tf.Variable(self.build_mat([self.image_feature_dim, self.hidden_dim]))
            self.UHigh_manager = tf.Variable(self.build_mat([self.emb_dim, self.hidden_dim]))
            params.extend([self.WHigh_manager, self.UHigh_manager])
            def unit(hidden_memory_tuple, focus, prev_x):
                hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
                # hidden_state : batch x hidden_dim
                hidden_state += tf.matmul(focus, self.WHigh_manager) + tf.matmul(prev_x, self.UHigh_manager)
                hidden_state = tf.nn.tanh(hidden_state)
                logits = tf.matmul(hidden_state, self.Wo) + self.bo
                # output = tf.nn.softmax(logits)
                return logits

            return unit
