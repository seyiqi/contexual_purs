import pickle
import os, time, random, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import tensorflow as tf
from model import Model
from sklearn.metrics import roc_auc_score
import tqdm
import sys

from train_purs import *

def produce_representations(sess, model, dataset, metafeature_dict, phase):
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
    all_representations = []
    all_items_embed = []


    epoch_size = round(len(dataset) / batch_size)
    dataloder = DataInput(dataset, batch_size) 
    if len(metafeature_dict) > 0:
        dataloder.metafeature_dict = metafeature_dict

    start_time = time.time()

    for batch, uij in dataloder:
        score, label, user, item, unexp, representations, items_embed = model.eval_saving_representations(sess, uij)

        all_scores.append(score)
        all_labels.append(label)
        all_users.append(user)
        all_items.append(item)
        all_exp.append(unexp)
        all_representations.append(representations)
        all_items_embed.append(items_embed)

        timenow = time.time() - start_time
        sys.stdout.write("\r\t Eval: %s %.2fs/step Step %d/%d" %
                             (phase, timenow/batch, batch, epoch_size))
        sys.stdout.flush()

    all_scores = [y for x in all_scores for y in x]
    all_labels = [y>0 for x in all_labels for y in x]
    all_users = [y for x in all_users for y in x]
    all_items = [y for x in all_items for y in x]
    all_exp = [y for x in all_exp for y in x]
    all_representations = [y for x in all_representations for y in x]
    all_items_embed = [y for x in all_items_embed for y in x]
    return all_scores, all_labels, all_users, all_items, all_exp, all_representations, all_items_embed

if __name__ == "__main__":

    # arg parse to override default setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=625)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="/cpu:0")
    parser.add_argument("--dataset", type=str, default="jester_small")
    parser.add_argument("--with-meta-data", action='store_true')
    parser.add_argument("--embeddingsize", type=int, default=50)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)

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

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # start experiment
    gpu_options = tf.GPUOptions(allow_growth=True)
    config=tf.ConfigProto(allow_soft_placement=True,  log_device_placement=True, gpu_options=gpu_options )

    with tf.Session(config=config) as sess:
        # model set up
        model = Model(user_count, item_count, batch_size, 
            metafeaturesize= len(list(metafeature_dict.items())[0][1]) if len(metafeature_dict) > 0 else 0,
            datatype=args.dataset,
            device=args.device)

        print(os.path.join(args.model_path, 'best_val.model'))
        model.restore(sess, path=os.path.join(args.model_path, 'best_val.model'))
        print("model loaded.")

        lr = 0.0
        last_auc = 0.0
        highest_auc = 0.0

        epoch_num = 1

        # evaluation
        start_time = time.time()
        model.global_epoch_step_op.eval()
        #eval_saving_representations
        
        ts_preds, ts_labels, ts_usrs, ts_items, ts_exps, ts_representations, item_embedding = produce_representations(sess, 
            model, test_set, metafeature_dict, 'test')
        # calculate stats
        ts_stats = report_model(ts_preds, ts_labels, ts_usrs, ts_items, ts_exps, item_count)

        # report stats
        print('\nEpoch %d DONE\tCost time: %.2f' %
              (model.global_epoch_step.eval(), time.time() - start_time))
        print('Epoch %d'%model.global_epoch_step.eval(), 'test stats: {}'.format(ts_stats))
        # update stats

        records = dict(zip(['pred', 'label', 'user', 'item', 'exp', 'representation', 'item_embedding'], 
            [ts_preds, ts_labels, ts_usrs, ts_items, ts_exps, ts_representations, item_embedding]))
        
        pickle.dump(records, 
            open(os.path.join(args.save_path, 'records.pkl'), "wb" ))






