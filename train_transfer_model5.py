# without using the traditional feature : sequence length. For shared representation, we use the hidden layer in Pyramid
# define the output layer using W and b
import tensorflow as tf
import numpy as np
import sys

from preprocess3 import SNLI
from Pyramid_Model5 import Pyramid_Model5
from utils import build_path
from sklearn import linear_model, svm
from sklearn.externals import joblib
from sklearn import metrics

RandSeed = 1234
def test_accuracy(pred, true):
    if len(pred) != len(true):
        print("error: the length of two lists are not the same")
        return 0
    else:
        count = 0
        for i in range(len(pred)):
            if pred[i] == true[i]:
                count += 1
        return float(count)/len(pred)

def F_score1(pred, true):
    if len(pred) != len(true):
        print("error: the length of two lists are not the same")
        return 0
    else:
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(len(pred)):
            if pred[i] == true[i] and pred[i] == 1:
                TP += 1
            elif pred[i] == true[i] and pred[i] == 0:
                TN += 1
            elif pred[i] != true[i] and pred[i] == 1:
                FP += 1
            elif pred[i] != true[i] and pred[i] == 0:
                FN += 1
        if (TP+FP) == 0:
            precision = 0
        else:
            precision = float(TP)/(TP+FP)
        if (TP+FN) == 0:
            precision = 0
        else:
            recall = float(TP)/(TP+FN)
        if precision ==0 or recall ==0:
            F_score =0
        else:
            F_score = 2*precision*recall/(precision+recall)
        return F_score

def re_norm(A):
    B = A.copy()
    for i in range(A.shape[0]):
        B[i][i] = 1.0
        for j in range(i+1, A.shape[0]):
            B[i][j] = B[i][j]/np.sqrt(np.abs(B[i][i]*B[j][j]))
            B[j][i] = B[i][j]
    return B

def norm_v(x, mu, sigma):
    #score = 1/(sigma*def_v) * np.exp(-(x-mu)**2 / (2 * sigma**2))
    score = np.exp(-(x-mu)**2 / (2 * sigma**2))
    return score

def print_matrix(A, meta):
    print (meta,)
    assert A.ndim == 2
    for i in range(len(A)):
        outstr = ""
        for v in A[i,:]:
            outstr += "%.3f "%v
        print ("  " + outstr.strip())
    print

def gen_vocab(data_list):
    word_to_idx = {}
    idx = 1
    for data_type in data_list:
        for i in range(data_type.data_size):
            s1 = data_type.s1s[i]
            for word in s1:
                if word not in word_to_idx:
                    word_to_idx[word] = idx
                    idx += 1
            s2 = data_type.s2s[i]
            for word in s2:
                if word not in word_to_idx:
                    word_to_idx[word] = idx
                    idx += 1
    return word_to_idx


def gen_trained_word_embedding(word2id):
    embeddings_index = {}
    f = open('../../glove.840B.300d.txt', 'r')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    np.random.seed(RandSeed)
    embedding_matrix = np.random.uniform(-0.01, 0.01, (len(word2id)+1, 300))
    embedding_matrix[0] = 0
    # embedding_matrix = np.zeros((len(self.word2id), self.word_embed_size))
    vocab_size = len(word2id)
    pretrained_size = 0
    for word, i in word2id.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            pretrained_size += 1
            embedding_matrix[i] = embedding_vector

    print('vocab size:%d\t pretrained size:%d' % (vocab_size, pretrained_size))

    return embedding_matrix

def train(lr, w, l2_reg, epoch, batch_size, model_type, num_layers, data_type, num_classes=3, genre = "telephone"):
    src_train_data = SNLI()
    tgt_train_data = SNLI()
    test_data = SNLI()

    src_train_data.open_file(mode="train", domain="src", genre = "snli")
    tgt_train_data.open_file(mode="train", domain="tgt", genre = genre)
    test_data.open_file(mode="test", domain="tgt", genre = genre)

    data_list = [src_train_data, tgt_train_data, test_data]
    word2idx = gen_vocab(data_list)
    fout = open('word_index', 'w')
    for word, i in word2idx.items():
        fout.write(word+'\t'+str(i)+'\n')
    fout.close()

    embedding_weight = gen_trained_word_embedding(word2idx)

    max_len = max(src_train_data.max_len, tgt_train_data.max_len)

    src_total_s1, src_total_s2 = src_train_data.gen_data(word2idx=word2idx, max_len = max_len)
    tgt_total_s1, tgt_total_s2 = tgt_train_data.gen_data(word2idx=word2idx, max_len = max_len)
    test_total_s1, test_total_s2 = test_data.gen_data(word2idx=word2idx, max_len=max_len)

    print("=" * 50)
    print("src training data size:", src_train_data.data_size)
    print("training max len:", max_len)
    print("=" * 50)

    model = Pyramid_Model5(s=max_len, w=w, l2_reg=l2_reg, model_type=model_type, embeddings = embedding_weight,
                  num_features=src_train_data.num_features,
                  num_classes=num_classes, num_layers=num_layers, corr_w = 0.5)

    src_optimizer = tf.train.AdagradOptimizer(lr, name="optimizer").minimize(model.src_cost)
    tgt_optimizer = tf.train.AdagradOptimizer(lr, name="optimizer").minimize(model.tgt_cost)
    #corr_optimizer = tf.train.AdagradOptimizer(lr, name="optimizer").minimize(model.corr_cost)

    task_no = 4
    update_corr = True

    # Assume all tasks are unrelated
    feed_site_corr = np.identity(task_no, dtype='float')/task_no
    sigma_feed_site_corr = np.identity(task_no, dtype='float') / task_no

    print ('feed_site_corr min %.3f, max %.3f'%(feed_site_corr.min(),\
                                                feed_site_corr.max()))
    feed_site_corr = np.linalg.pinv(feed_site_corr)
    print ('feed_site_corr min %.3f, max %.3f'%(feed_site_corr.min(),\
                                                feed_site_corr.max()))

    # Due to GTX 970 memory issues
    gpu_options = tf.GPUOptions()

    # Initialize all variables
    init = tf.global_variables_initializer()

    # model(parameters) saver
    saver = tf.train.Saver(max_to_keep=100)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        ##train_summary_writer = tf.summary.FileWriter("C:/tf_logs/train", sess.graph)
        '''
        model_path = build_path("./models/", data_type, model_type, num_layers)
        print(model_path + "-" + str(1))
        print('load sess')
        saver.restore(sess, model_path + "-" + str(5))
        '''
        sess.run(init)
        acc_list = []

        print("=" * 50)
        for e in range(1, epoch + 1):
            print("[Epoch " + str(e) + "]")

            src_list = list(zip(src_total_s1, src_total_s2, src_train_data.labels, src_train_data.aux_labels))
            tgt_list = list(zip(tgt_total_s1, tgt_total_s2, tgt_train_data.labels, tgt_train_data.aux_labels))

            np.random.shuffle(src_list)
            np.random.shuffle(tgt_list)

            new_src_total_s1, new_src_total_s2, new_src_label, new_src_aux_label = [], [], [], []
            new_tgt_total_s1, new_tgt_total_s2, new_tgt_label, new_tgt_aux_label = [], [], [], []

            for s_tuple in src_list:
                new_src_total_s1.append(s_tuple[0])
                new_src_total_s2.append(s_tuple[1])
                new_src_label.append(s_tuple[2])
                new_src_aux_label.append(s_tuple[3])

            for s_tuple in tgt_list:
                new_tgt_total_s1.append(s_tuple[0])
                new_tgt_total_s2.append(s_tuple[1])
                new_tgt_label.append(s_tuple[2])
                new_tgt_aux_label.append(s_tuple[3])

            new_src_total_s1 = np.asarray(new_src_total_s1)
            new_src_total_s2 = np.asarray(new_src_total_s2)
            new_src_label = np.asarray(new_src_label)
            new_src_aux_label = np.asarray(new_src_aux_label)
            new_tgt_total_s1 = np.asarray(new_tgt_total_s1)
            new_tgt_total_s2 = np.asarray(new_tgt_total_s2)
            new_tgt_label = np.asarray(new_tgt_label)
            new_tgt_aux_label = np.asarray(new_tgt_aux_label)

            src_train_data.reset_index()
            tgt_train_data.reset_index()
            i = 0

            while src_train_data.is_available():
                i += 1

                src_batch_x1, src_batch_x2, src_batch_y, src_batch_features, src_batch_aux = src_train_data.next_batch(
                    new_src_total_s1, new_src_total_s2, labels=new_src_label, aux_labels=new_src_aux_label,
                    batch_size=batch_size)

                if not tgt_train_data.is_available():
                    tgt_train_data.reset_index()

                    # for updating the correlatio matrix
                    if update_corr:
                        if e <= -1:
                            FeedCorr = 0  # 0:skip update, 1:MT (1/2), 2:SGD (no tr=1)
                        else:
                            FeedCorr = 1
                        # update feed_site_corr
                        if FeedCorr == 0:
                            print("Skip update!")
                        else:
                            site_corr_trans, m_w = sess.run( \
                                [model.site_corr_trans, model.weights],
                                feed_dict={model.x1: tgt_batch_x1,
                                           model.x2: tgt_batch_x2,
                                           model.y: tgt_batch_y})
                            if FeedCorr == 1:
                                # this part is for the update of omega without logdet
                                if not np.isnan(np.sum(m_w)):
                                    U, s, V = np.linalg.svd(site_corr_trans)
                                    # S = np.sqrt(np.abs(np.diag(s)))
                                    S = np.sqrt(np.diag(s))
                                    A = np.dot(U, np.dot(S, V))
                                    A = A / np.trace(A) # this is the covariance matrix
                                    # print ('  feed_site_corr A(no inv)\n', A)
                                    print_matrix(A, '  feed_site_corr A(no inv)')
                                    print('  feed_site_corr trace', np.trace(A))
                                    # site_corr_new = np.linalg.pinv(A)
                                    # renorm A
                                    B = re_norm(A) # B is the correlation matrix
                                    # print ('  feed_site_corr B(no inv)\n', B)
                                    print_matrix(B, '  feed_site_corr B(no inv)')
                                    site_corr_new = np.linalg.pinv(A) # the inverse of the covariance matrix
                                else:
                                    site_corr_new = np.nan
                                    print('m_w nan, skip!!')
                            else:
                                # this part is for the update of omega with logdet, and it may not work
                                site_corr_new = np.linalg.inv(site_corr_trans + \
                                                              sigma_feed_site_corr)
                                # trick
                                site_corr_new = site_corr_new / np.trace(site_corr_new)
                            if np.isnan(np.sum(site_corr_new)):
                                print(' site_corr_new nan, skipped!')
                            else:
                                feed_site_corr = site_corr_new
                            # print (' site_corr_trans', site_corr_trans)
                            if m_w.shape[1] == 1:
                                print(' m_w\t' + " ".join(["%.5f" % (v) for v in m_w]))
                        pass
                    pass

                    # for testing
                    test_data.reset_index()
                    predict_score = []
                    predict_value = []
                    true_score = []
                    QA_pairs = {}
                    # labels = test_data.labels
                    s1s, s2s, labels, features = test_data.next_batch2(test_total_s1, test_total_s2,
                                                                      labels=test_data.labels,
                                                                      batch_size=test_data.data_size)
                    for i in range(test_data.data_size):
                        pred, clf_input = sess.run([model.tgt_prediction, model.tgt_output_features],
                                                   feed_dict={model.x1: np.expand_dims(s1s[i], axis=0),
                                                              model.x2: np.expand_dims(s2s[i], axis=0),
                                                              model.y: np.expand_dims(labels[i], axis=0),
                                                              model.features: np.expand_dims(features[i], axis=0)}
                                                   )

                        true_score.append(labels[i])
                        predict_score.append(np.argmax(pred))
                        # print(len(QA_pairs.keys()))
                    test_acc = test_accuracy(predict_score, true_score)
                    print('acc' + str(test_acc))
                    acc_list.append(test_acc)
                    print("current max acc=" + str(max(acc_list)))

                tgt_batch_x1, tgt_batch_x2, tgt_batch_y, tgt_batch_features, tgt_batch_aux = tgt_train_data.next_batch(
                    new_tgt_total_s1, new_tgt_total_s2, labels=new_tgt_label, aux_labels=new_tgt_aux_label,
                    batch_size=batch_size)

                '''
                src_batch_x1, src_batch_x2, src_batch_y, src_batch_features = src_train_data.next_batch(src_total_s1,
                                                                                                        src_total_s2, batch_size = batch_size)

                if not tgt_train_data.is_available():
                    tgt_train_data.reset_index()
                tgt_batch_x1, tgt_batch_x2, tgt_batch_y, tgt_batch_features = tgt_train_data.next_batch(
                    tgt_total_s1, tgt_total_s2, batch_size=batch_size)
                '''

                _, sc = sess.run([src_optimizer, model.src_cost],
                                                  feed_dict={model.x1: src_batch_x1,
                                                             model.x2: src_batch_x2,
                                                             model.y: src_batch_y,
                                                             model.aux: src_batch_aux,
                                                             model.site_corr: feed_site_corr,
                                                             model.features: src_batch_features})

                _, tc = sess.run([tgt_optimizer, model.tgt_cost],
                                                  feed_dict={model.x1: tgt_batch_x1,
                                                             model.x2: tgt_batch_x2,
                                                             model.y: tgt_batch_y,
                                                             model.aux: tgt_batch_aux,
                                                             model.site_corr: feed_site_corr,
                                                             model.features: tgt_batch_features})
                
                #if e > 5:
                   # _, cc = sess.run([corr_optimizer, model.corr_cost],
                    #                              feed_dict={model.site_corr: feed_site_corr})
                #else:
                    #cc = 0
                                                            
                if i % 100 == 0:
                    print("[batch " + str(i) + "] src cost:", sc)
                    print("[batch " + str(i) + "] tgt cost:", tc)
                    #_, cc = sess.run([corr_optimizer, model.corr_cost],
                     #                             feed_dict={model.site_corr: feed_site_corr})
                    #print("[batch " + str(i) + "] corr cost:", cc)
                    
                #train_summary_writer.add_summary(merged, i)

            save_path = saver.save(sess, build_path("./models/", data_type, model_type, num_layers), global_step=e)
            print("model saved as", save_path)

        print("training finished!")
        print("=" * 50)


if __name__ == "__main__":

    # Paramters
    # --lr: learning rate
    # --ws: window_size
    # --l2_reg: l2_reg modifier
    # --epoch: epoch
    # --batch_size: batch size
    # --model_type: model type
    # --num_layers: number of convolution layers

    # default parameters

    genre = "telephone"

    params = {
        "lr": 0.08,
        "ws": 4,
        "l2_reg": 0.0004,
        "epoch": 5,
        "batch_size": 64,
        "model_type": "Model51",
        "num_layers": 1,
        "data_type": "AliExp"
    }

    print("=" * 50)
    print("Parameters:")
    for k in sorted(params.keys()):
        print(k, ":", params[k])

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            k = arg.split("=")[0][2:]
            v = arg.split("=")[1]
            params[k] = v

    train(lr=float(params["lr"]), w=int(params["ws"]), l2_reg=float(params["l2_reg"]), epoch=int(params["epoch"]),
          batch_size=int(params["batch_size"]), model_type=params["model_type"], num_layers=int(params["num_layers"]),
          data_type=params["data_type"], genre = genre)
