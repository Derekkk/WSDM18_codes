import tensorflow as tf
import numpy as np


class Pyramid_Model3():
    def __init__(self, s, w, l2_reg, model_type, embeddings, num_features, d0=300, di=150, d_att1 = 8, d_att2 = 16, w_att1 = 6, w_att2 = 4, num_classes=3, num_layers=2, corr_w = 0.1):

        self.x1 = tf.placeholder(tf.int32, shape=[None, s], name="x1")
        self.x2 = tf.placeholder(tf.int32, shape=[None, s], name="x2")
        self.y = tf.placeholder(tf.int32, shape=[None], name="y")
        self.features = tf.placeholder(tf.float32, shape=[None, num_features], name="features")
        l2_reg_lambda = l2_reg
        last_output_layer_size = di * 4 + d_att2 * (s/4/3/2) * (s/4/3/2)

        self.corr_w = corr_w

        self.E = tf.Variable(embeddings, trainable=True, dtype=tf.float32)

        emb1_ori = tf.nn.embedding_lookup(self.E, self.x1)
        emb2_ori = tf.nn.embedding_lookup(self.E, self.x2)

        emb1 = tf.transpose(emb1_ori, [0, 2, 1], name="emb1_trans")
        emb2 = tf.transpose(emb2_ori, [0, 2, 1], name="emb2_trans")

        # zero padding to inputs for wide convolution
        def pad_for_wide_conv(x):
            return tf.pad(x, np.array([[0, 0], [0, 0], [w - 1, w - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

        def cos_sim(v1, v2):
            norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
            norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
            dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

            return dot_products / (norm1 * norm2)

        def euclidean_score(v1, v2):
            euclidean = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1))
            return 1 / (1 + euclidean)

        def make_attention_mat(x1, x2):
            # x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]
            # x2 => [batch, height, 1, width]
            # [batch, width, wdith] = [batch, s, s]
            #euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1))
            dot = tf.reduce_sum(tf.matmul(x1,tf.matrix_transpose(x2)), axis=1, name="att_sim")
            return dot

        def convolution(name_scope, x, d):
            # Convolution layer for BCNN
            with tf.name_scope(name_scope + "-conv"):
                with tf.variable_scope("conv") as scope:
                    conv = tf.contrib.layers.conv2d(
                        inputs=x,
                        num_outputs=di,
                        kernel_size=(d, w),
                        stride=1,
                        padding="VALID",
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                        biases_initializer=tf.constant_initializer(1e-04),
                        reuse=True,
                        trainable=True,
                        scope=scope
                    )
                    # Weight: [filter_height, filter_width, in_channels, out_channels]
                    # output: [batch, 1, input_width+filter_Width-1, out_channels] == [batch, 1, s+w-1, di]

                    # [batch, di, s+w-1, 1]
                    conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
                    return conv_trans

        def convolution1(name_scope, variable_scope, x, kernel_vector, d_output):
            with tf.name_scope(name_scope + "-conv1"):
                with tf.variable_scope(variable_scope) as scope:
                    conv = tf.contrib.layers.conv2d(
                        inputs=x,
                        num_outputs=d_output,
                        kernel_size=kernel_vector,
                        stride=1,
                        padding="SAME",
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                        biases_initializer=tf.constant_initializer(1e-04),
                        reuse=False,
                        trainable=True,
                        scope=scope
                    )
                    # Input: [batch, s, s, 1]
                    # Weight: [filter_height, filter_width, in_channels, out_channels]
                    # output: [batch, s-d+1, s-d+1, d_output]

                    # [batch, di, s+w-1, 1]
                    #conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
                    # [batch, s-d+1, s-d+1, d_output]
                    conv_trans = conv
                    return conv_trans

        def convolution2(name_scope, variable_scope, x, kernel_vector, d_output):
            with tf.name_scope(name_scope + "-conv2"):
                with tf.variable_scope(variable_scope) as scope:
                    conv = tf.contrib.layers.conv2d(
                        inputs=x,
                        num_outputs=d_output,
                        kernel_size=kernel_vector,
                        stride=3,
                        padding="VALID",
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                        biases_initializer=tf.constant_initializer(1e-04),
                        reuse=False,
                        trainable=True,
                        scope=scope
                    )
                    # Input: [batch, s, s, 1]
                    # Weight: [filter_height, filter_width, in_channels, out_channels]
                    # output: [batch, s-d+1, s-d+1, d_output]

                    # [batch, di, s+w-1, 1]
                    #conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
                    # [batch, s-d+1, s-d+1, d_output]
                    conv_trans = conv
                    return conv_trans

        def w_pool(variable_scope, x, attention):
            # Window Pooling layer for BCNN (if necessary, this is used as the first layer)
            # x: [batch, di, s+w-1, 1]
            # attention: [batch, s+w-1]
            with tf.variable_scope(variable_scope + "-w_pool"):
                if model_type == "ABCNN2" or model_type == "ABCNN3":
                    pools = []
                    # [batch, s+w-1] => [batch, 1, s+w-1, 1]
                    attention = tf.transpose(tf.expand_dims(tf.expand_dims(attention, -1), -1), [0, 2, 1, 3])

                    for i in range(s):
                        # [batch, di, w, 1], [batch, 1, w, 1] => [batch, di, 1, 1]
                        pools.append(tf.reduce_sum(x[:, :, i:i + w, :] * attention[:, :, i:i + w, :],
                                                   axis=2,
                                                   keep_dims=True))

                    # [batch, di, s, 1]
                    w_ap = tf.concat(pools, axis=2, name="w_ap")
                else:
                    w_ap = tf.layers.average_pooling2d(
                        inputs=x,
                        # (pool_height, pool_width)
                        pool_size=(1, w),
                        strides=1,
                        padding="VALID",
                        name="w_ap"
                    )
                    # [batch, di, s, 1]

                return w_ap

        def all_att_pool1(variable_scope, x, pool_vector):
            with tf.variable_scope(variable_scope + "-all_pool"):
                all_ap = tf.layers.max_pooling2d(
                    inputs=x,
                    # (pool_height, pool_width)
                    pool_size=pool_vector,
                    strides=4,
                    padding="VALID",
                    name="all_ap"
                )
                # [batch, di, 1, 1]

                return all_ap

        def all_att_pool2(variable_scope, x, pool_vector, d_hidden):
            with tf.variable_scope(variable_scope + "-all_pool"):
                all_ap = tf.layers.max_pooling2d(
                    inputs=x,
                    # (pool_height, pool_width)
                    pool_size=pool_vector,
                    strides=2,
                    padding="VALID",
                    name="all_ap"
                )
                # [batch, di, 1, 1]

                # [batch, di]
                all_ap_reshaped = tf.reshape(all_ap, [-1, d_hidden*(s/4/3/2)*(s/4/3/2)])
                #all_ap_reshaped = tf.squeeze(all_ap, [2, 3])

                return all_ap_reshaped

        def all_pool(variable_scope, x):
            # All Pooling Layer for BCNN
            with tf.variable_scope(variable_scope + "-all_pool"):
                if variable_scope.startswith("input"):
                    pool_width = s
                    d = d0
                    all_ap = tf.layers.average_pooling2d(
                        inputs=x,
                        # (pool_height, pool_width)
                        pool_size=(1, pool_width),
                        strides=1,
                        padding="VALID",
                        name="all_ap"
                    )
                else:
                    pool_width = s + w - 1
                    d = di
                    all_ap = tf.layers.max_pooling2d(
                        inputs=x,
                        # (pool_height, pool_width)
                        pool_size=(1, pool_width),
                        strides=1,
                        padding="VALID",
                        name="all_ap"
                    )
                # [batch, di, 1, 1]

                # [batch, di]
                all_ap_reshaped = tf.reshape(all_ap, [-1, d])
                #all_ap_reshaped = tf.squeeze(all_ap, [2, 3])

                return all_ap_reshaped

        def CNN_layer(variable_scope, x1, x2, d):
            # x1, x2 = [batch, d, s, 1]
            with tf.variable_scope(variable_scope):
                with tf.name_scope("att_mat"):
                    # [batch, s, s]
                    att_mat = make_attention_mat(x1, x2)

                att_mat_expanded = tf.expand_dims(att_mat, -1)
                left_conv = convolution(name_scope="left", x=pad_for_wide_conv(x1), d=d)
                right_conv = convolution(name_scope="right", x=pad_for_wide_conv(x2), d=d)

                left_attention, right_attention = None, None

                left_wp = w_pool(variable_scope="left", x=left_conv, attention=left_attention)
                left_ap = all_pool(variable_scope="left", x=left_conv)
                right_wp = w_pool(variable_scope="right", x=right_conv, attention=right_attention)
                right_ap = all_pool(variable_scope="right", x=right_conv)

                print(att_mat_expanded.shape)
                att_conv = convolution1(name_scope="att", variable_scope="att", x=att_mat_expanded, kernel_vector=(w_att1,w_att1), d_output=d_att1) #[batch, s-w_att1+1, s-w_att1+1, d_att1]
                att_ap = all_att_pool1(variable_scope="att", x=att_conv, pool_vector=(4,4)) #[batch, s-2*w_att1+2, s-2*w_att1+2, d_att1]
                print(att_ap.shape)
                att_conv2 = convolution2(name_scope="att2", variable_scope="att2", x=att_ap, kernel_vector=(w_att2,w_att2), d_output=d_att2)
                att_ap2 = all_att_pool2(variable_scope="att2", x=att_conv2, pool_vector=(2,2), d_hidden=d_att2)

                return left_wp, left_ap, right_wp, right_ap, att_ap2

        x1_expanded = tf.expand_dims(emb1, -1)
        x2_expanded = tf.expand_dims(emb2, -1)

        src_LI_1, src_LO_1, src_RI_1, src_RO_1, src_att_ap = CNN_layer(variable_scope="src_CNN-1", x1=x1_expanded, x2=x2_expanded, d=d0)
        #src_sims = [cos_sim(LO_0, RO_0)]
        src_diff = tf.subtract(src_LO_1, src_RO_1)
        src_mul = tf.multiply(src_LO_1, src_RO_1)

        LI_1, LO_1, RI_1, RO_1, att_ap = CNN_layer(variable_scope="CNN-1", x1=x1_expanded, x2=x2_expanded, d=d0)
        #sims = [cos_sim(LO_0, RO_0)]
        diff = tf.subtract(LO_1, RO_1)
        mul = tf.multiply(LO_1, RO_1)

        tgt_LI_1, tgt_LO_1, tgt_RI_1, tgt_RO_1, tgt_att_ap = CNN_layer(variable_scope="tgt_CNN-1", x1=x1_expanded, x2=x2_expanded, d=d0)
        #tgt_sims = [cos_sim(LO_0, RO_0)]
        tgt_diff = tf.subtract(tgt_LO_1, tgt_RO_1)
        tgt_mul = tf.multiply(tgt_LO_1, tgt_RO_1)
        #sims = [cos_sim(LO_0, RO_0)]

        self.src_output_features = tf.concat([src_att_ap, src_LO_1, src_RO_1, src_diff, src_mul], axis=1, name="src_output_features")

        self.shared_output_features = tf.concat([att_ap, LO_1, RO_1, diff, mul], axis=1, name="shared_output_features")

        self.tgt_output_features = tf.concat([tgt_att_ap, tgt_LO_1, tgt_RO_1, tgt_diff, tgt_mul], axis=1, name="tgt_output_features")

        with tf.variable_scope("shared-src-output-layer"):
            self.shared_src_estimation = tf.contrib.layers.fully_connected(
                inputs=self.shared_output_features,
                num_outputs=num_classes,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

        with tf.variable_scope("shared-tgt-output-layer"):
            self.shared_tgt_estimation = tf.contrib.layers.fully_connected(
                inputs=self.shared_output_features,
                num_outputs=num_classes,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

        with tf.variable_scope("src-output-layer"):
            self.src_estimation = tf.contrib.layers.fully_connected(
                inputs=self.src_output_features,
                num_outputs=num_classes,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

        with tf.variable_scope("tgt-output-layer"):
            self.tgt_estimation = tf.contrib.layers.fully_connected(
                inputs=self.tgt_output_features,
                num_outputs=num_classes,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

        self.src_prediction = tf.contrib.layers.softmax(self.src_estimation + self.shared_src_estimation)[:, :]

        self.src_pred_cost = tf.add(
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.src_estimation + self.shared_src_estimation, labels=self.y)),
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
            name="src_cost")

        self.tgt_prediction = tf.contrib.layers.softmax(self.tgt_estimation + self.shared_tgt_estimation)[:, :]

        self.tgt_pred_cost = tf.add(
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.tgt_estimation + self.shared_tgt_estimation, labels=self.y)),
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
            name="tgt_cost")

        with tf.variable_scope("src-output-layer", reuse = True):
            src_W_s = tf.get_variable("FC/weights")
        with tf.variable_scope("shared-src-output-layer", reuse = True):
            src_W_c = tf.get_variable("FC/weights")
        with tf.variable_scope("shared-tgt-output-layer", reuse = True):
            tgt_W_c = tf.get_variable("FC/weights")
        with tf.variable_scope("tgt-output-layer", reuse = True):
            tgt_W_t = tf.get_variable("FC/weights")

        src_W_s_reshaped = tf.reshape(src_W_s, [-1, last_output_layer_size * num_classes])
        src_W_c_reshaped = tf.reshape(src_W_c, [-1, last_output_layer_size * num_classes])
        tgt_W_t_reshaped = tf.reshape(tgt_W_t, [-1, last_output_layer_size * num_classes])
        tgt_W_c_reshaped = tf.reshape(tgt_W_c, [-1, last_output_layer_size * num_classes])

        # weight matrix: 4*H, H=2+d_att2
        self.weights = tf.concat([src_W_s_reshaped, src_W_c_reshaped, tgt_W_t_reshaped, tgt_W_c_reshaped], 0)

        # to model the correlation between weights
        self.site_corr = tf.placeholder('float', [4, 4])
        trans_w = tf.transpose(self.weights)
        self.corr2 = tf.matmul(trans_w, tf.matmul(self.site_corr, self.weights))
        # self.corr2 = tf.matmul(tf.matmul(trans_w, self.site_corr), self.weights['out'])
        print('Corr2', self.corr2)
        self.corr_trace = tf.trace(self.corr2)
        print('Corr_trace', self.corr_trace)

        # for updating site_corr
        self.site_corr_trans = tf.matmul(self.weights, trans_w)

        # 0.001 is the lambda
        self.src_cost = self.src_pred_cost + tf.reduce_sum(l2_reg_lambda*self.corr_trace*self.corr_w)

        # 0.001 is the lambda
        self.tgt_cost = self.tgt_pred_cost + tf.reduce_sum(l2_reg_lambda*self.corr_trace*self.corr_w)

        #self.corr_cost = tf.reduce_sum(l2_reg_lambda*self.corr_trace*self.corr_w)

        tf.summary.scalar("cost", self.src_cost)
        tf.summary.scalar("cost", self.tgt_cost)
        self.merged = tf.summary.merge_all()

        print("=" * 50)
        print("List of Variables:")
        for v in tf.trainable_variables():
            print(v.name)
        print("=" * 50)
