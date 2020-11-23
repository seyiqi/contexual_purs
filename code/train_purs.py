import os, time, random, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import tensorflow as tf
from model import Model
from sklearn.metrics import roc_auc_score
import tqdm
import sys

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
        self.metafeature_dict={}

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
        u, hist, i, y, meta = [], [], [], [], []
        for t in ts:
            u.append(t[0])
            hist.append(t[1])
            i.append(t[2])
            y.append(t[3])
            if t[2] in self.metafeature_dict:
                meta.append(self.metafeature_dict[t[2]])
        return self.i, (u, hist, i, y, meta)

class Stats:
    def __init__(self, cols):
        self.stats_dict = dict([(x, []) for x in cols])

    def update_stats(self, data_dict, epoch, phase):
        assert len(data_dict) == (len(self.stats_dict)-2)
        for col in data_dict:
            self.stats_dict[col].append(data_dict[col])
        self.stats_dict["epoch"].append(epoch)
        self.stats_dict["phase"].append(phase)

    def to_csv(self, csv_dir):
        pd.DataFrame(self.stats_dict).to_csv(csv_dir)


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


def post_process_data(batch_size, data):
    tr_df, val_df, ts_df = data

    # populate history
    train_history = calculate_history(tr_df)
    val_history = calculate_history(val_df)
    test_history = calculate_history(ts_df)
    train_data_mat = list(tr_df[['user_id', 'video_id', 'click']].values)
    val_data_mat = list(val_df[['user_id', 'video_id', 'click']].values)
    test_data_mat = list(ts_df[['user_id', 'video_id', 'click']].values)
    train_set = [(train_data_mat[i][0], train_history[i], train_data_mat[i][1], train_data_mat[i][2]) for i in
                 range(len(train_data_mat))]
    validation_set = [(val_data_mat[i][0], val_history[i], val_data_mat[i][1], val_data_mat[i][2]) for i in
                 range(len(val_data_mat))]
    test_set = [(test_data_mat[i][0], test_history[i], test_data_mat[i][1], test_data_mat[i][2]) for i in
                range(len(test_data_mat))]

    # post processing
    # TODO: this is very ugly and not a reasonable thing to do!
    train_set = train_set[:len(train_set) // batch_size * batch_size]
    val_set = validation_set[:len(validation_set) // batch_size * batch_size]
    test_set = test_set[:len(test_set) // batch_size * batch_size]
    return train_set, val_set, test_set


def load_dataset(data_dir, post_fix=""):
    """
    Function that loads the dataset from a directory
    :param data_dir:
    :param post_fix:
    :return:
    """
    tr_df = pd.read_csv(os.path.join(data_dir, "train{}.csv".format(post_fix)))
    val_df = pd.read_csv(os.path.join(data_dir, "validation{}.csv".format(post_fix)))
    ts_df = pd.read_csv(os.path.join(data_dir, "test{}.csv".format(post_fix)))
    return tr_df, val_df, ts_df


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


def load_jester_data(batch_size, postfix):
    # rename the dimensions
    tr_df, val_df, ts_df = load_dataset('../data/jester/clean/', postfix)
    tr_df = tr_df.rename(columns={"user":"user_id", "item":"video_id", "binary_y":"click"})
    val_df = val_df.rename(columns={"user":"user_id", "item":"video_id", "binary_y":"click"})
    ts_df = ts_df.rename(columns={"user":"user_id", "item":"video_id", "binary_y":"click"})
    all_df = pd.concat([tr_df, val_df, ts_df])
    user_count = len(all_df["user_id"].unique())
    item_count = len(all_df["video_id"].unique())

    # add random hour column
    tr_df["hour"] = np.random.randint(low=0, high=23, size=len(tr_df))
    val_df["hour"] = np.random.randint(low=0, high=23, size=len(val_df))
    ts_df["hour"] = np.random.randint(low=0, high=23, size=len(ts_df))
    tr_df = tr_df[['user_id', 'video_id', 'click', 'hour']]
    val_df = val_df[['user_id', 'video_id', 'click', 'hour']]
    ts_df = ts_df[['user_id', 'video_id', 'click', 'hour']]

    # post processing and split the dataset
    tr, val, ts  = post_process_data(batch_size, [tr_df, val_df, ts_df])
    return tr, val, ts, user_count, item_count

def get_name_embeddings(data_ori, col, embeddingsize):
    name_vocabulary = [] 
    for name in data_ori[col].values:
        name_vocabulary.extend(" ".join(name.split()).lower().split(' ')) 

    name_vocabulary = list(set(name_vocabulary))
    vocab = dict(zip(name_vocabulary,  np.random.randn(len(name_vocabulary), embeddingsize)))

    name_embeddings = np.zeros((len(data_ori), embeddingsize))
    for i, name in enumerate(data_ori[col].values):
        words = " ".join(name.split()).lower().split(' ')

        for word in words:
            name_embeddings[i]+=vocab[word]
    return name_embeddings, vocab

def load_beer_metadata(data_ori, unique_items, embeddingsize=100):
    #data_ori = pd.read_csv('../data/beer/clean_beer_reviews.csv', index_col = 0)

    data_ori = data_ori.drop(['review_time', 'review_overall', 
                              'review_aroma', 'review_appearance', 
                              'review_profilename', 'review_palate', 'review_taste', 'reviewer_id', 
                              #'num_tasted_beers'
                              ], axis=1)

    data_ori = data_ori.drop_duplicates()
    data_ori = data_ori.set_index('beer_beerid')

    for i in unique_items:
        if i not in data_ori.index:
            print(i)
    data_ori = data_ori.loc[unique_items]
    data_ori.beer_abv = data_ori.beer_abv.fillna(0)
    data_ori.brewery_name = data_ori.brewery_name.fillna('unknown')

    name_embeddings, name_vocab = get_name_embeddings(data_ori, 'beer_name', embeddingsize)
    brewery_name_embeddings, brewery_name_vocab = get_name_embeddings(data_ori, 'brewery_name', embeddingsize)
    print('Unique words in beer name: ', len(name_vocab))
    print('Unique words in brewery name: ', len(brewery_name_vocab))

    features = np.concatenate([name_embeddings, 
                                    brewery_name_embeddings, 
                                    np.expand_dims(data_ori.style_id.values, axis=1), 
                                    np.expand_dims(data_ori.beer_abv.values, axis=1), 
                                   ], 1)

    feature_dict = dict(zip(data_ori.index, features))

    return feature_dict


def load_beer_data(batch_size, postfix, with_meta_data=False, embeddingsize=100):
    # load and preprocess data
    # data = pd.read_csv('../data/beer/small_clean_beer_reviews.csv')

    tr_df, val_df, ts_df = load_dataset('../data/beer', postfix)
    all_df_withmeta = pd.concat([tr_df, val_df, ts_df])
    #print(all_df_withmeta.columns)

    tr_df = tr_df.rename(columns={"reviewer_id":"user_id", "beer_beerid":"video_id","review_time":"hour", "binary_y":"click"})
    val_df = val_df.rename(columns={"reviewer_id":"user_id", "beer_beerid":"video_id","review_time":"hour", "binary_y":"click"})
    ts_df = ts_df.rename(columns={"reviewer_id":"user_id", "beer_beerid":"video_id","review_time":"hour", "binary_y":"click"})
    all_df = pd.concat([tr_df, val_df, ts_df])
    user_count = len(all_df["user_id"].unique())
    item_count = len(all_df["video_id"].unique())
    tr_df = tr_df[['user_id', 'video_id', 'click', 'hour']]
    val_df = val_df[['user_id', 'video_id', 'click', 'hour']]
    ts_df = ts_df[['user_id', 'video_id', 'click', 'hour']]

    # post processing and split the dataset
    tr, val, ts = post_process_data(batch_size, [tr_df, val_df, ts_df])

    metafeature_dict = {}
    if with_meta_data:
        # print(list(all_df["video_id"].unique()))
        metafeature_dict = load_beer_metadata(all_df_withmeta, list(all_df["video_id"].unique()), embeddingsize)

    return tr, val, ts, user_count, item_count, metafeature_dict


def eval_model(sess, model, dataset, metafeature_dict):
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

    dataloder = DataInput(dataset, batch_size) 
    if len(metafeature_dict) > 0:
        dataloder.metafeature_dict = metafeature_dict

    for _, uij in dataloder:
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


def report_model(preds, labels, usrs, items, exps, item_count):
    """
    Give a full report including multiple metrics for a set of predictions
    :param labels:
    :param preds:
    :param usrs:
    :param items:
    :param exps:
    :param item_count:
    :return:
    """
    # auc
    auc_score = roc_auc_score(labels, preds)
    # hit
    hit = hit_rate(preds, labels, usrs)
    # coverage
    cov = coverage(preds, items, item_count)
    # unexpectedness
    unexp = np.nanmean(exps)
    return {"auc":auc_score, "hit":hit, "cov":cov, "unexp":unexp}


if __name__ == "__main__":

    # arg parse to override default setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=625)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epoch", type=int, default=100)
    parser.add_argument("--device", type=str, default="/cpu:0")
    parser.add_argument("--dataset", type=str, default="jester_small")
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--stats-path", type=str, default=None)
    parser.add_argument("--with-meta-data", action='store_true')
    parser.add_argument("--embeddingsize", type=int, default=50)
    args = parser.parse_args()

    # experiment set up
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    batch_size = args.batch_size

    metafeature_dict = {}

    # load dataset
    if args.dataset == "jester":
        train_set, val_set, test_set, user_count, item_count = load_jester_data(batch_size, "")
    elif args.dataset == "jester_small":
        train_set, val_set, test_set, user_count, item_count = load_jester_data(batch_size, "_small")
    elif args.dataset == "beer":
        train_set, val_set, test_set, user_count, item_count, metafeature_dict = load_beer_data(batch_size, "", 
            args.with_meta_data, args.embeddingsize)
    elif args.dataset == "beer_small":
        train_set, val_set, test_set, user_count, item_count, metafeature_dict = load_beer_data(batch_size, "_small", 
            args.with_meta_data, args.embeddingsize)
    elif args.dataset == "beer_ultrasmall":
        train_set, val_set, test_set, user_count, item_count, metafeature_dict = load_beer_data(batch_size, "_ultrasmall", 
            args.with_meta_data, args.embeddingsize)
    else:
        raise ValueError("Invalid dataset option {}".format(args.dataset))
    print("data loaded.")

    # start experiment
    stats_holder = Stats(["auc", "hit", "cov", "unexp", "epoch", "phase"])
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # model set up
        model = Model(user_count, item_count, batch_size, 
            metafeaturesize= len(list(metafeature_dict.items())[0][1]) if len(metafeature_dict) > 0 else 0,
            datatype=args.dataset)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        lr = args.learning_rate
        last_auc = 0.0

        # epoch loop
        for epoch_num in range(1, 1000):
            # training
            start_time = time.time()
            random.shuffle(train_set)
            epoch_size = round(len(train_set) / batch_size)
            loss_sum = 0.0
            dataloder = DataInput(train_set, batch_size) 
            if len(metafeature_dict) > 0:
                dataloder.metafeature_dict = metafeature_dict

            for batch, uij in dataloder:
                loss = model.train(sess, uij, lr)
                loss_sum += loss
                timenow = time.time() - start_time
                sys.stdout.write("\rEpoch %d/%d %.2fs/step Step %d/%d: %s= %f" %
                             (epoch_num, 1000, timenow/batch, batch, epoch_size, 'loss', loss_sum/batch))
                sys.stdout.flush()

            # evaluation
            model.global_epoch_step_op.eval()
            tr_preds, tr_labels, tr_usrs, tr_items, tr_exps = eval_model(sess, model, train_set, metafeature_dict)
            val_preds, val_labels, val_usrs, val_items, val_exps = eval_model(sess, model, val_set, metafeature_dict)
            ts_preds, ts_labels, ts_usrs, ts_items, ts_exps = eval_model(sess, model, test_set, metafeature_dict)
            # calculate stats
            tr_stats = report_model(tr_preds, tr_labels, tr_usrs, tr_items, tr_exps, item_count)
            val_stats = report_model(val_preds, val_labels, val_usrs, val_items, val_exps, item_count)
            ts_stats = report_model(ts_preds, ts_labels, ts_usrs, ts_items, ts_exps, item_count)
            # report stats
            print('Epoch %d DONE\tCost time: %.2f' %
                  (model.global_epoch_step.eval(), time.time() - start_time))
            print('Epoch %d loss: %.4f' % (model.global_epoch_step.eval(), loss_sum))
            print('Epoch %d'%model.global_epoch_step.eval(), 'training stats: {}'.format(tr_stats))
            print('Epoch %d'%model.global_epoch_step.eval(), 'validation stats: {}'.format(val_stats))
            print('Epoch %d'%model.global_epoch_step.eval(), 'test stats: {}'.format(ts_stats))
            # update stats
            stats_holder.update_stats(tr_stats, epoch_num, "training")
            stats_holder.update_stats(val_stats, epoch_num, "validation")
            stats_holder.update_stats(ts_stats, epoch_num, "test")
            if args.stats_path is not None:
                stats_holder.to_csv(args.stats_path)

            # adjust learning rate
            train_auc = tr_stats["auc"]
            if abs(train_auc - last_auc) < 0.001:
                lr /= 2
            last_auc = train_auc