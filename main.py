import sys
import xgboost as xgb
import pandas as pd


def run():
    dtrain = xgb.DMatrix("./data/mq2008.train")
    dtrain_group = load_group_file("./data/mq2008.train.group")
    dtrain.set_group(dtrain_group)

    dvali = xgb.DMatrix("./data/mq2008.vali")
    dvali_group = load_group_file("./data/mq2008.vali.group")
    dvali.set_group(dvali_group)

    dtest = xgb.DMatrix("./data/mq2008.test")
    dtest_group = load_group_file("./data/mq2008.test.group")
    dtest.set_group(dtest_group)

    params = {"objective": "rank:pairwise",
            "eta": 0.1,
            "gamma": 1.0,
            "min_child_weight": 1,
            "max_depth": 6}

    watchlist = [(dvali, 'eval'), (dtrain, 'train')]
    num_round = 4
    bst = xgb.train(params, dtrain, num_round, watchlist)

    # predict
    preds = bst.predict(dtest)
    print '\nPrediction size:', len(preds)
    for index, pred in enumerate(preds):
        print index + 1, pred
    
def load_group_file(file_path):
    group = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                group.append(int(line.strip()))
            except Exception as ex:
                print "Exception happen at line:", line

    return group

if __name__ == "__main__":
    run()