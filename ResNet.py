import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data


"""
CIFAR-10 데이터셋 처리코드와 next_batch 함수사용을 다음 링크에서 참고했습니다.
https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data
http://solarisailab.com/archives/2325
"""

class ResNet:
    def __init__(self):
        self.input_size = None
        self.lr = None
        self.wd = None
        self.momentum = None
        self.done_epoch = None
        self.sampling_step = None

        ###손실함수, 정확도 그래프를 그리기 위해
        self.metric_list = {}
        self.metric_list['losses'] = []
        self.metric_list['test_acc'] = []
        self.metric_list['train_acc'] = []

        self.graph = tf.Graph()

    ###########################논문에서는 CIFAR-10의 경우 보틀넥도, 프로젝션 숏컷도 사용하지 않았지만
    ###########################이 실험에서는 각각의 사용유무에 따른 4개의 모델을 만들어 놓고 실험결과를 보도록 한다.
    def make_residual_bottleneck_block(self, input, ch, is_training, downsampling = False):
        W1 = tf.Variable(tf.truncated_normal([1, 1, ch, ch//4], stddev=tf.sqrt(2/ch)), dtype=np.float32)
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W1), self.wd))
        L1 = tf.nn.conv2d(input, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.layers.batch_normalization(L1, training=is_training)
        L1 = tf.nn.relu(L1)


        if (downsampling) :
            W2 = tf.Variable(tf.truncated_normal([3, 3, ch//4, ch//4 * 2], stddev=tf.sqrt(2/(ch//4))), dtype=np.float32)
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W2), self.wd))
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 2, 2, 1], padding='SAME')
            L2 = tf.layers.batch_normalization(L2, training=is_training)
            L2 = tf.nn.relu(L2)

            W3 = tf.Variable(tf.truncated_normal([1, 1, ch//4 * 2, ch * 2], stddev=tf.sqrt(2/(ch//4 * 2))), dtype=np.float32)
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W3), self.wd))
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
            L3 = tf.layers.batch_normalization(L3, training=is_training)

            ###Apply Projection to skip connection When DownSample
            W_projection = tf.Variable(tf.truncated_normal([1, 1, ch, ch * 2], stddev=tf.sqrt(2/ch)), dtype=np.float32)
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W_projection), self.wd))
            L_projection = tf.nn.conv2d(input, W_projection, strides = [1, 2, 2, 1],
                                        padding='SAME') #DownSample by strided 1x1 Convolution

            L4 = tf.add(L3, L_projection)

        else :
            W2 = tf.Variable(tf.truncated_normal([3, 3, ch // 4, ch // 4], stddev=tf.sqrt(2/(ch//4))), dtype=np.float32)
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W2), self.wd))
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.layers.batch_normalization(L2, training=is_training)
            L2 = tf.nn.relu(L2)

            W3 = tf.Variable(tf.truncated_normal([1, 1, ch // 4, ch], stddev=tf.sqrt(2/(ch//4))), dtype=np.float32)
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W3), self.wd))
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
            L3 = tf.layers.batch_normalization(L3, training=is_training)

            L4 = tf.add(L3, input)

        L4 = tf.nn.relu(L4)

        return L4

    def make_residual_block(self, input, ch, is_training, downsampling=False):
        W1 = tf.Variable(tf.truncated_normal([3, 3, ch, ch], stddev=tf.sqrt(2/ch)), dtype=np.float32)
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W1), self.wd))
        L1 = tf.nn.conv2d(input, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.layers.batch_normalization(L1, training=is_training)
        L1 = tf.nn.relu(L1)

        if (downsampling):
            W2 = tf.Variable(tf.truncated_normal([3, 3, ch, ch * 2], stddev=tf.sqrt(2/ch)), dtype=np.float32)
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W2), self.wd))
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 2, 2, 1], padding='SAME')
            L2 = tf.layers.batch_normalization(L2, training=is_training)

            ###Apply Projection to skip connection When DownSample
            W_projection = tf.Variable(tf.truncated_normal([1, 1, ch, ch * 2], stddev=tf.sqrt(2/ch)), dtype=np.float32)
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W_projection), self.wd))
            L_projection = tf.nn.conv2d(input, W_projection, strides=[1, 2, 2, 1],
                                        padding='SAME')  # DownSample by strided 1x1 Convolution

            L3 = tf.add(L2, L_projection)

        else:
            W2 = tf.Variable(tf.truncated_normal([3, 3, ch, ch], stddev=tf.sqrt(2/ch)), dtype=np.float32)
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W2), self.wd))
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.layers.batch_normalization(L2, training=is_training)

            L3 = tf.add(L2, input)

        L3 = tf.nn.relu(L3)

        return L3

    def make_residual_block_downsample_with_zero_padding(self, input, ch, is_training):
        W1 = tf.Variable(tf.truncated_normal([3, 3, ch, ch], stddev=tf.sqrt(2/ch)), dtype=np.float32)
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W1), self.wd))
        L1 = tf.nn.conv2d(input, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.layers.batch_normalization(L1, training=is_training)
        L1 = tf.nn.relu(L1)

        W2 = tf.Variable(tf.truncated_normal([3, 3, ch, ch * 2], stddev=tf.sqrt(2/ch)), dtype=np.float32)
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W2), self.wd))
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 2, 2, 1], padding='SAME')
        L2 = tf.layers.batch_normalization(L2, training=is_training)

        ###Apply Zero-Padding to skip connection When DownSample
        input = tf.nn.max_pool(input, ksize=[1, 1, 1, 1], strides=[1, 2, 2, 1], padding='SAME')
        L_zero_padding = tf.pad(input, [[0, 0], [0, 0], [0, 0], [ch//2, ch//2]])

        L3 = tf.add(L2, L_zero_padding)

        L3 = tf.nn.relu(L3)

        return L3

    def block_repeat(self, L_input, func_, ch, is_training, iter):
        for _ in range(iter) :
            L_input = func_(L_input, ch, is_training)
        return L_input

    def build1(self, input, label, is_training=False):   #########보틀넥 x 프로젝션 숏컷x (논문의 구현에 가장 충실한 모델)
        W_input = tf.Variable(tf.truncated_normal([3, 3, 3, 16], stddev=1), dtype=np.float32)
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W_input), self.wd))
        L_input = tf.nn.conv2d(input, W_input, strides=[1, 1, 1, 1], padding='SAME')
        L_input = tf.layers.batch_normalization(L_input, training=is_training)
        L_input = tf.nn.relu(L_input)

        block1 = self.block_repeat(L_input, self.make_residual_block, 16, is_training, 9 - 1)
        block2 = self.make_residual_block_downsample_with_zero_padding(block1, 16, is_training)

        block3 = self.block_repeat(block2, self.make_residual_block, 32, is_training, 9 - 1)
        block4 = self.make_residual_block_downsample_with_zero_padding(block3, 32, is_training)

        block5 = self.block_repeat(block4, self.make_residual_block, 64, is_training, 9)

        GAP = tf.reduce_mean(block5, [1, 2], keep_dims=False)
        # GAP = tf.reduce_mean(block5, [1, 2], keepdims=False)

        W_fc = tf.Variable(tf.truncated_normal([64, 10], stddev=0.1), dtype=np.float32)
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W_fc), self.wd))

        logit = tf.matmul(GAP, W_fc)

        prediction = tf.argmax(logit, axis=1)

        # res_block1 = self.make_residual_block(L_input, 16, is_training)
        # res_block2 = self.make_residual_block(res_block1, 16, is_training)
        # res_block3 = self.make_residual_block(res_block2, 16, is_training)
        # res_block4 = self.make_residual_block(res_block3, 16, is_training)
        # res_block5 = self.make_residual_block(res_block4, 16, is_training)
        # res_block6 = self.make_residual_block(res_block5, 16, is_training)
        # res_block7 = self.make_residual_block(res_block6, 16, is_training)
        # res_block8 = self.make_residual_block(res_block7, 16, is_training)
        # res_block9 = self.make_residual_block_downsample_with_zero_padding(res_block8, 16, is_training)
        #
        # res_block10 = self.make_residual_block(res_block9, 32, is_training)
        # res_block11 = self.make_residual_block(res_block10, 32, is_training)
        # res_block12 = self.make_residual_block(res_block11, 32, is_training)
        # res_block13 = self.make_residual_block(res_block12, 32, is_training)
        # res_block14 = self.make_residual_block(res_block13, 32, is_training)
        # res_block15 = self.make_residual_block(res_block14, 32, is_training)
        # res_block16 = self.make_residual_block(res_block15, 32, is_training)
        # res_block17 = self.make_residual_block(res_block16, 32, is_training)
        # res_block18 = self.make_residual_block_downsample_with_zero_padding(res_block17, 32, is_training)
        #
        # res_block19 = self.make_residual_block(res_block9, 32, is_training)
        # res_block20 = self.make_residual_block(res_block10, 32, is_training)
        # res_block21 = self.make_residual_block(res_block11, 32, is_training)
        # res_block22 = self.make_residual_block(res_block12, 32, is_training)
        # res_block23 = self.make_residual_block(res_block13, 32, is_training)
        # res_block24 = self.make_residual_block(res_block14, 32, is_training)
        # res_block25 = self.make_residual_block(res_block15, 32, is_training)
        # res_block26 = self.make_residual_block(res_block16, 32, is_training)
        # res_block27 = self.make_residual_block_downsample_with_zero_padding(res_block17, 32, is_training)

        return prediction, logit

    def build2(self, input, label, is_training=False):  #########보틀넥 x 프로젝션 숏컷o
        W_input = tf.Variable(tf.truncated_normal([3, 3, 3, 16], stddev=1), dtype=np.float32)
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W_input), self.wd))
        L_input = tf.nn.conv2d(input, W_input, strides=[1, 1, 1, 1], padding='SAME')
        L_input = tf.layers.batch_normalization(L_input, training=is_training)
        L_input = tf.nn.relu(L_input)

        block1 = self.block_repeat(L_input, self.make_residual_block, 16, is_training, 9 - 1)
        block2 = self.make_residual_block(block1, 16, is_training, downsampling=True)

        block3 = self.block_repeat(block2, self.make_residual_block, 32, is_training, 9 - 1)
        block4 = self.make_residual_block(block3, 32, is_training, downsampling=True)

        block5 = self.block_repeat(block4, self.make_residual_block, 64, is_training, 9)

        GAP = tf.reduce_mean(block5, [1, 2], keep_dims=False)
        # GAP = tf.reduce_mean(block5, [1, 2], keepdims=False)

        W_fc = tf.Variable(tf.truncated_normal([64, 10], stddev=0.1), dtype=np.float32)
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W_fc), self.wd))

        logit = tf.matmul(GAP, W_fc)

        prediction = tf.argmax(logit, axis=1)

        return prediction, logit

    def build3(self, input, label, is_training=False):  #########보틀넥 o 프로젝션 숏컷x
        W_input = tf.Variable(tf.truncated_normal([3, 3, 3, 16], stddev=1), dtype=np.float32)
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W_input), self.wd))
        L_input = tf.nn.conv2d(input, W_input, strides=[1, 1, 1, 1], padding='SAME')
        L_input = tf.layers.batch_normalization(L_input, training=is_training)
        L_input = tf.nn.relu(L_input)

        block1 = self.block_repeat(L_input, self.make_residual_bottleneck_block, 16, is_training, 9 - 1)
        block2 = self.make_residual_block_downsample_with_zero_padding(block1, 16, is_training)

        block3 = self.block_repeat(block2, self.make_residual_bottleneck_block, 32, is_training, 9 - 1)
        block4 = self.make_residual_block_downsample_with_zero_padding(block3, 32, is_training)

        block5 = self.block_repeat(block4, self.make_residual_bottleneck_block, 64, is_training, 9)

        GAP = tf.reduce_mean(block5, [1, 2], keep_dims=False)
        # GAP = tf.reduce_mean(block5, [1, 2], keepdims=False)

        W_fc = tf.Variable(tf.truncated_normal([64, 10], stddev=0.1), dtype=np.float32)
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W_fc), self.wd))

        logit = tf.matmul(GAP, W_fc)

        prediction = tf.argmax(logit, axis=1)

        return prediction, logit

    def build4(self, input, label, is_training=False):  #########보틀넥 o 프로젝션 숏컷o
        W_input = tf.Variable(tf.truncated_normal([3, 3, 3, 16], stddev=1), dtype=np.float32)
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W_input), self.wd))
        L_input = tf.nn.conv2d(input, W_input, strides=[1, 1, 1, 1], padding='SAME')
        L_input = tf.layers.batch_normalization(L_input, training=is_training)
        L_input = tf.nn.relu(L_input)

        block1 = self.block_repeat(L_input, self.make_residual_bottleneck_block, 16, is_training, 9 - 1)
        block2 = self.make_residual_bottleneck_block(block1, 16, is_training, downsampling=True)

        block3 = self.block_repeat(block2, self.make_residual_bottleneck_block, 32, is_training, 9 - 1)
        block4 = self.make_residual_bottleneck_block(block3, 32, is_training, downsampling=True)

        block5 = self.block_repeat(block4, self.make_residual_bottleneck_block, 64, is_training, 9)

        GAP = tf.reduce_mean(block5, [1, 2], keep_dims=False)
        # GAP = tf.reduce_mean(block5, [1, 2], keepdims=False)

        W_fc = tf.Variable(tf.truncated_normal([64, 10], stddev=0.1), dtype=np.float32)
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W_fc), self.wd))

        logit = tf.matmul(GAP, W_fc)

        prediction = tf.argmax(logit, axis=1)

        return prediction, logit

    def train(self, input, label, is_training, model_kind=1):
        if model_kind == 1:
            pred, logit_ = self.build1(input, label, is_training)
        elif model_kind == 2:
            pred, logit_ = self.build2(input, label, is_training)
        elif model_kind == 3:
            pred, logit_ = self.build3(input, label, is_training)
        else:
            pred, logit_ = self.build4(input, label, is_training)

        CEE = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(logits=logit_, labels=label)))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(CEE), self.wd))

        loss = tf.add_n(tf.get_collection('losses'))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = tf.train.MomentumOptimizer(self.lr, self.momentum).minimize(loss)

        ###### L2-loss가 더해진, Training Operation에 사용되는 진짜 loss는
        ###### 패러미터 수에 따라 그 값이 커지기 때문에 모델간 성능 비교를 위해선 loss대신 CEE(Cross Entropy Error)를 반환한다.
        return train_op, pred, CEE, logit_

    def next_batch (self, num, data, label) :
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        label_shuffle = [label[i] for i in idx]

        return np.asarray(data_shuffle), np.asarray(label_shuffle)

    def test_sort(self, start, end, data, labels):
        data_shuffle = data[start:end]
        labels_shuffle = labels[start:end]

        return np.asarray(data_shuffle), np.asarray(labels_shuffle)


    def run(self, max_iter, model_kind, input_size, lr, wd, momentum, done_epoch, sampling_step):
        self.input_size = input_size
        self.lr = lr
        self.wd = wd
        self.momentum = momentum
        self.done_epoch = done_epoch
        self.sampling_step = sampling_step

        with self.graph.as_default():
            ###cifar 10 로드
            (x_train, y_train), (x_test, y_test) = load_data()

            y_train = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
            y_test = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

            X = tf.placeholder(tf.float32, [None, x_train.shape[1], x_train.shape[2], x_train.shape[3]])
            Y = tf.placeholder(tf.float32, [None, 10])
            is_training = tf.placeholder(tf.bool)

            train_op, pred, loss, logit = self.train(X, Y, is_training, model_kind)
            correct_prediction = tf.equal(pred, tf.argmax(Y, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            sess = tf.Session()

            saver = tf.train.Saver(tf.global_variables())

            model_folder = './model' + str(model_kind)
            ckpt = tf.train.get_checkpoint_state(model_folder)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())


            data_size = y_train.shape[0].value
            batch_num = data_size // self.input_size

            start_time = time.time()
            for epoch in range(max_iter) :
                for itr in range(batch_num):
                    input_batch, label_batch = self.next_batch(self.input_size, x_train, y_train.eval(session=sess))
                    _, loss_ = sess.run([train_op, loss], feed_dict={X: input_batch, Y: label_batch, is_training: True})

                    if itr % 10 == 0:
                        progress_view = 'progress : ' + '%7.6f'%(itr / batch_num * 100) + '%  loss :' + '%7.6f'%loss_
                        print(progress_view)
                        self.metric_list['losses'].append(loss_)

                with open('loss.txt', 'a') as wf:
                    epoch_time = time.time() - start_time
                    loss_info = '\nepoch: ' + '%7d' % (
                                epoch + 1 + self.done_epoch) + '  batch loss: ' + '%7.6f' % loss_ + '  time elapsed: ' + '%7.6f'%epoch_time
                    wf.write(loss_info)

                if epoch % self.sampling_step == 0:
                    test_accuracy = 0

                    start_test_time = time.time()
                    for i in range(x_test.shape[0] // 100):
                        input_batch, label_batch = self.test_sort(100 * i, 100 * (i + 1), x_test, y_test.eval(session=sess))
                        tmpacc = sess.run(accuracy, feed_dict={X: input_batch, Y: label_batch, is_training: False})
                        test_accuracy = test_accuracy + tmpacc / (x_test.shape[0] // 100)

                    inference_time_test = time.time() - start_test_time

                    train_accuracy = 0
                    start_test_time = time.time()
                    for i in range(x_train.shape[0] // 100):
                        input_batch, label_batch = self.test_sort(100 * i, 100 * (i + 1), x_train,
                                                                  y_train.eval(session=sess))
                        tmpacc = sess.run(accuracy, feed_dict={X: input_batch, Y: label_batch, is_training: False})
                        train_accuracy = train_accuracy + tmpacc / (x_train.shape[0] // 100)

                    inference_time_train = time.time() - start_test_time

                    self.reg_acc(test_accuracy, train_accuracy, inference_time_test, inference_time_train)

                    if epoch % 20 == 0 :
                        model_dir = './model'+str(model_kind)+ '_epoch'+ str(epoch+1+self.done_epoch) +'/model.ckpt'
                        saver.save(sess, model_dir)

            model_dir = './model' + str(model_kind) + '/model.ckpt'
            saver.save(sess, model_dir)
            sess.close()


    def inference (self, model_kind) :
        with self.graph.as_default():
            ###cifar 10 로드
            (x_train, y_train), (x_test, y_test) = load_data()

            y_train = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
            y_test = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

            X = tf.placeholder(tf.float32, [None, x_train.shape[1], x_train.shape[2], x_train.shape[3]])
            Y = tf.placeholder(tf.float32, [None, 10])
            is_training = tf.placeholder(tf.bool)

            train_op, pred, loss, logit = self.train(X, Y, is_training, model_kind)
            correct_prediction = tf.equal(pred, tf.argmax(Y, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            sess = tf.Session()

            saver = tf.train.Saver(tf.global_variables())

            model_folder = './model' + str(model_kind)
            ckpt = tf.train.get_checkpoint_state(model_folder)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())


            test_accuracy = 0

            start_test_time = time.time()
            for i in range(x_test.shape[0] // 100):
                input_batch, label_batch = self.test_sort(100 * i, 100 * (i + 1), x_train, y_train.eval(session=sess))
                tmpacc = sess.run(accuracy, feed_dict={X: input_batch, Y: label_batch, is_training: False})
                test_accuracy = test_accuracy + tmpacc / (x_test.shape[0] // 100)

            print('test accuracy %g' % test_accuracy)
            test_time = time.time() - start_test_time
            print('test time %g' % test_time)

    def save_acc(self):
        x = range(0, len(self.metric_list['test_acc']), self.sampling_step)

        y1 = self.metric_list['test_acc']
        y2 = self.metric_list['train_acc']

        plt.plot(x, y1, label='train_acc')
        plt.plot(x, y2, label='test_acc')

        plt.xlabel('Epoch')
        plt.ylabel('Acc')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        plt.savefig('acc.png')

        plt.close()

    def save_loss(self):
        x = range(0, len(self.metric_list['losses']), self.sampling_step)

        y1 = self.metric_list['losses']

        plt.plot(x, y1, label='loss')

        plt.xlabel('Iter')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        plt.savefig('loss.png')

        plt.close()

    def reg_acc(self, test_accuracy, train_accuracy, inference_time_test, inference_time_train) :
        print('test accuracy %g' % test_accuracy)
        self.metric_list['test_acc'].append(test_accuracy)
        with open('loss.txt', 'a') as wf:
            acc = '\ntest accuracy: ' + '%7g' % test_accuracy + '   inference_time: %7.6f' % inference_time_test
            wf.write(acc)

        print('train accuracy %g' % train_accuracy)
        self.metric_list['test_acc'].append(train_accuracy)
        with open('loss.txt', 'a') as wf:
            acc = '\ntrain accuracy: ' + '%7g' % train_accuracy + '   inference_time: %7.6f' % inference_time_train
            wf.write(acc)