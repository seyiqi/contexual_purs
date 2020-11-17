import os, sys, time, random, argparse, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import pandas as pd
import numpy as np
import tensorflow as tf
from model import Model
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from lenskit.crossfold import partition_users, SampleFrac

# Note: this code must be run using tensorflow 1.4.0

class DataInput:
    """
    The ugly dataloader from the author
    """
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
        """
        Ugly!
        :return:
        """
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


def eval_model(sess, model, dataset):
    """
    Function that run inference for a model on a dataset
    :param sess:
    :param model:
    :param dataset:
    :return:
    """
    all_scores = []
    all_labels = []
    all_users = []
    all_items = []
    all_exp = []
    for _, uij in DataInput(dataset, batch_size):
        score, label, user, item, unexp = model.test(sess, uij)
        all_scores.append(score)
        all_labels.append(label)
        all_users.append(user)
        all_items.append(item)
        all_exp.append(unexp)
    all_scores = [y for x in all_scores for y in x]
    all_labels = [y>0 for x in all_labels for y in x]
    all_users = [y for x in all_users for y in x]
    all_items = [y for x in all_items for y in x]
    all_exp = [y for x in all_exp for y in x]
    return all_scores, all_labels, all_users, all_items, all_exp


def hit_rate(preds, labels, users, topk=10):
    user_pred_dict = {}
    hit_rates = []
    for i in range(len(preds)):
        if users[i] not in user_pred_dict:
            user_pred_dict[users[i]] = []
        user_pred_dict[users[i]].append((preds[i], labels[i]))
    for user in user_pred_dict:
        user_res = sorted(user_pred_dict[user], key=lambda x: x[0])[-topk:]
        hit_rates.append(np.sum([x[1] > 0 for x in user_res])/topk)
    return np.mean(hit_rates)


def coverage(preds, items, num_items):
    rec_item = []
    for i in range(len(preds)):
        if preds[i] > 0.5:
            rec_item.append(items[i])
    return len(set(rec_item)) / num_items


def calculate_history(df):
    """
    Create the history for each data point
    Note that history for each data point needs to be exactly 10 elements (UGLY!)
    :param df:
    :return:
    """
    df = df.sort_values(["user_id", "hour"])
    output = []
    current_user = None
    current_list = []
    user_col = list(df["user_id"])
    item_col = list(df["video_id"])
    for i in range(len(df)):
        if current_user != user_col[i]:
            current_list = []
            current_user = user_col[i]
        current_list.append(item_col[i])
        # if history has more than 10 elements, throw out the first element
        if len(current_list) > 10:
            current_list = current_list[1:]
        # if history has less than 10 elements, pad it with the last element
        to_be_added = current_list.copy()
        if len(to_be_added) < 10:
            to_be_added += [to_be_added[-1] for _ in range(10-len(to_be_added))]
        output.append(to_be_added)
    return output


def post_process_data(batch_size, data, seed):
    # train test split
    res = list(partition_users(data.rename(columns={"user_id": "user"}), 1, SampleFrac(0.2), rng_spec=seed))[0]
    train_data = res.train.rename(columns={"user": "user_id"})
    test_data = res.test.rename(columns={"user": "user_id"})

    # populate history
    train_history = calculate_history(train_data)
    test_history = calculate_history(test_data)
    train_data_mat = list(train_data[['user_id', 'video_id', 'click']].values)
    test_data_mat = list(test_data[['user_id', 'video_id', 'click']].values)
    train_set = [(train_data_mat[i][0], train_history[i], train_data_mat[i][1], train_data_mat[i][2]) for i in
                 range(len(train_data_mat))]
    test_set = [(test_data_mat[i][0], test_history[i], test_data_mat[i][1], test_data_mat[i][2]) for i in
                range(len(test_data_mat))]

    # post processing
    random.shuffle(train_set)
    random.shuffle(test_set)
    # TODO: this is very ugly and not a reasonable thing to do!
    train_set = train_set[:len(train_set) // batch_size * batch_size]
    test_set = test_set[:len(test_set) // batch_size * batch_size]
    return train_set, test_set


def load_purs_data(batch_size, seed):
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


def load_jester_data(batch_size, seed):
    # rename the dimensions
    data = pd.read_csv('../data/jester/clean/ratings.csv').rename(columns={"user":"user_id", "item":"video_id"})
    # TODO: it takes very long time to procss all reviews (7.2M) so we only use 10% of it
    data = data.head(720000)
    user_count = len(data["user_id"].unique())
    item_count = len(data["video_id"].unique())
    # binarize the labels
    data["click"] = (data["rating"] > 0).astype(int)
    # add random hour column
    data["hour"] = np.random.randint(low=0, high=23, size=len(data))
    data = data[['user_id', 'video_id', 'click', 'hour']]

    # post processing and split the dataset
    train_set, test_set = post_process_data(batch_size, data, seed)

    return train_set, test_set, user_count, item_count


def load_beer_data(batch_size, seed):
    # load and preprocess data
    data = pd.read_csv('../data/beer/small_clean_beer_reviews.csv').rename(columns={"reviewer_id":"user_id", "beer_beerid":"video_id",
                                                                                    "review_time":"hour"})
    # debug!
    #data = data.head(100000)

    # throw out users who has less than 10 comments
    data = data[data["num_tasted_beers"]>10]

    # reindex the user and item column
    le = preprocessing.LabelEncoder()
    data["user_id"] = le.fit_transform(data["user_id"])
    data["video_id"] = le.fit_transform(data["video_id"])
    user_count = len(data["user_id"].unique())
    item_count = len(data["video_id"].unique())

    # binarize the label
    data["click"] = (data["review_overall"] > 4).astype(int)
    data = data[['user_id', 'video_id', 'click', 'hour']]

    # post processing and split the dataset
    train_set, test_set = post_process_data(batch_size, data, seed)

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

    # load dataset
    if args.dataset == "purs":
        train_set, test_set, user_count, item_count = load_purs_data(batch_size, args.seed)
    elif args.dataset == "jester":
        train_set, test_set, user_count, item_count = load_jester_data(batch_size, args.seed)
    elif args.dataset == "beer":
        train_set, test_set, user_count, item_count = load_beer_data(batch_size, args.seed)
    else:
        raise ValueError("Invalid dataset option {}".format(args.dataset))
    print("data loaded.")

    # start experiment
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # model set up
        model = Model(user_count, item_count, batch_size)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        lr = 1
        last_auc = 0.0

        # epoch loop
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
            ts_preds, ts_labels, ts_usrs, ts_items, ts_exps = eval_model(sess, model, test_set)
            tr_preds, tr_labels, tr_usrs, tr_items, tr_exps = eval_model(sess, model, train_set)
            test_auc = roc_auc_score(ts_labels, ts_preds)
            train_auc = roc_auc_score(tr_labels, tr_preds)
            hit = hit_rate(ts_preds, ts_labels, ts_usrs)
            cov = coverage(ts_preds, ts_items, item_count)
            unexp = np.nanmean(ts_exps)
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