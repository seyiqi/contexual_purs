import os, sys, time, random, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import tensorflow as tf
from model import Model
from sklearn.metrics import roc_auc_score

# 1 use GPU: done
# 2 re-frame code: done
# 3 load jester data
# 4 load beer data


# Note: this code must be run using tensorflow 1.4.0

class DataInput:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1
        u, hist, i, y = [], [], [], []
        for t in ts:
            u.append(t[0])
            hist.append(t[1])
            i.append(t[2])
            y.append(t[3])
        return self.i, (u, hist, i, y)

def auc(sess, model, test_set):
    all_scores = []
    all_labels = []
    for _, uij in DataInput(test_set, batch_size):
        score, label, user, _, _ = model.test(sess, uij)
        all_scores.append(score)
        all_labels.append(label)
    all_scores = [y for x in all_scores for y in x]
    all_labels = [y>0 for x in all_labels for y in x]
    return roc_auc_score(all_labels, all_scores)


def hit_rate(sess, model, test_set):
    user_pred_dict = {}
    hit_rates = []
    for _, uij in DataInput(test_set, batch_size):
        score, label, user, _, _ = model.test(sess, uij)
        for i in range(len(score)):
            if user[i] not in user_pred_dict:
                user_pred_dict[user[i]] = []
            user_pred_dict[user[i]].append((score[i], label[i]))
    for user in user_pred_dict:
        user_res = sorted(user_pred_dict[user], key=lambda x: x[0])[-10:]
        hit_rates.append(np.sum([x[1] > 0 for x in user_res])/10)
    return np.mean(hit_rates)


def coverage(sess, model, test_set, num_items):
    rec_item = []
    for _, uij in DataInput(test_set, batch_size):
        score, label, user, item, _ = model.test(sess, uij)
        for index in range(len(score)):
            if score[index] > 0.5:
                rec_item.append(item[index])
    return len(set(rec_item)) / num_items


def unexpectedness(sess, model, test_set):
    unexp_list = []
    for _, uij in DataInput(test_set, batch_size):
        score, label, user, item, unexp = model.test(sess, uij)
        for index in range(len(score)):
            unexp_list.append(unexp[index])
    return np.mean(unexp_list)


def load_purs_data():
    # set up data
    data = pd.read_csv('../data/purs_data/test.txt', names=['utdid', 'vdo_id', 'click', 'hour'])
    # assign integer id to each user
    user_id = data[['utdid']].drop_duplicates().reindex()
    user_id['user_id'] = np.arange(len(user_id))
    data = pd.merge(data, user_id, on=['utdid'], how='left')
    # assign integer id to each item
    item_id = data[['vdo_id']].drop_duplicates().reindex()
    item_id['video_id'] = np.arange(len(item_id))
    data = pd.merge(data, item_id, on=['vdo_id'], how='left')
    data = data[['user_id', 'video_id', 'click', 'hour']]
    userid = list(set(data['user_id']))
    itemid = list(set(data['video_id']))
    user_count = len(userid)
    item_count = len(itemid)


    # train test split
    validate = 4 * len(data) // 5
    train_data = data.loc[:validate, ]
    test_data = data.loc[validate:, ]
    train_set, test_set = [], []

    for user in userid:
        train_user = train_data.loc[train_data['user_id'] == user]
        train_user = train_user.sort_values(['hour'])
        length = len(train_user)
        train_user.index = range(length)
        # we need at least 10 interactions from each user
        if length > 10:
            for i in range(length - 10):
                train_set.append((train_user.loc[i + 9, 'user_id'], list(train_user.loc[i:i + 9, 'video_id']),
                                  train_user.loc[i + 9, 'video_id'], float(train_user.loc[i + 9, 'click'])))

        test_user = test_data.loc[test_data['user_id'] == user]
        test_user = test_user.sort_values(['hour'])
        length = len(test_user)
        test_user.index = range(length)
        if length > 10:
            for i in range(length - 10):
                test_set.append((test_user.loc[i + 9, 'user_id'], list(test_user.loc[i:i + 9, 'video_id']),
                                 test_user.loc[i + 9, 'video_id'], float(test_user.loc[i + 9, 'click'])))
    random.shuffle(train_set)
    random.shuffle(test_set)
    train_set = train_set[:len(train_set) // batch_size * batch_size]
    test_set = test_set[:len(test_set) // batch_size * batch_size]
    return train_set, test_set, user_count, item_count




if __name__ == "__main__":

    # arg parse to override default setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=625)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epoch", type=int, default=100)
    parser.add_argument("--device", type=str, default="/cpu:0")
    parser.add_argument("--dataset", type=str, default="purs")
    args = parser.parse_args()

    # experiment set up
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    batch_size = args.batch_size

    # # load dataset
    if args.dataset == "purs":
        train_set, test_set, user_count, item_count = load_purs_data()

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = Model(user_count, item_count, batch_size)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        lr = 1
        last_auc = 0.0

        for _ in range(1000):
            # training
            start_time = time.time()
            random.shuffle(train_set)
            epoch_size = round(len(train_set) / batch_size)
            loss_sum = 0.0
            for _, uij in DataInput(train_set, batch_size):
                loss = model.train(sess, uij, lr)
                loss_sum += loss

            # evaluation
            model.global_epoch_step_op.eval()
            test_auc = auc(sess, model, test_set)
            train_auc = auc(sess, model, train_set)
            hit = hit_rate(sess, model, test_set)
            cov = coverage(sess, model, test_set, item_count)
            unexp = unexpectedness(sess, model, test_set)
            print('Epoch %d DONE\tCost time: %.2f' %
                  (model.global_epoch_step.eval(), time.time() - start_time))
            print('Epoch %d loss: %.4f' % (model.global_epoch_step.eval(), loss_sum))
            print('Epoch %d training auc: %.4f' % (model.global_epoch_step.eval(), train_auc))
            print('Epoch %d test auc: %.4f' % (model.global_epoch_step.eval(), test_auc))
            print('Epoch %d Eval_Hit_Rate: %.4f' % (model.global_epoch_step.eval(), hit))
            print('Epoch %d Eval_Coverage: %.4f' % (model.global_epoch_step.eval(), cov))
            print('Epoch %d Eval_Unexpectedness: %.4f' % (model.global_epoch_step.eval(), unexp))

            # adjust learning rate
            if abs(train_auc - last_auc) < 0.001:
                lr /= 2
            last_auc = train_auc