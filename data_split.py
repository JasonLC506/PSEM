import numpy as np
import cPickle

def train_test_split(posts, train_ratio, valid_ratio):
    """
    split reaction data into train, valid and test by ratio
    :param posts: {doc_id: [uids, eids]}
    :param train_ratio: percentage of training
    :param valid_ratio: percentage of validation
    :return: posts_train, posts_valid, posts_test
    """
    test_ratio = 1.0 - train_ratio - valid_ratio
    assert test_ratio > 0
    posts_train, posts_valid, posts_test = {}, {}, {}
    for doc_id in posts:
        uids, eids = posts[doc_id]
        uids = np.array(uids, dtype=np.int64)
        eids = np.array(eids, dtype=np.int64)

        Md = uids.shape[0]
        Md_train = max(1, int(Md * train_ratio))
        Md_valid = (Md - Md_train) * valid_ratio / (valid_ratio + test_ratio)
        Md_test = Md - Md_train - Md_valid

        inds = np.arange(Md)
        np.random.shuffle(inds)
        inds_train = inds[:Md_train]
        inds_valid = inds[Md_train: Md_train + Md_valid]
        inds_test = inds[Md_train + Md_valid: Md]

        uids_train, eids_train = uids[inds_train], eids[inds_train]
        uids_valid, eids_valid = uids[inds_valid], eids[inds_valid]
        uids_test, eids_test = uids[inds_test], eids[inds_test]

        ### test ###
        print("doc_id: %d, # train: %d, # valid: %d, # test: %d" % (doc_id, uids_train.shape[0], uids_valid.shape[0], uids_test.shape[0]))

        posts_train[doc_id] = [uids_train, eids_train]
        posts_valid[doc_id] = [uids_valid, eids_valid]
        posts_test[doc_id] = [uids_test, eids_test]
    return posts_train, posts_valid, posts_test


if __name__ == "__main__":
    posts = cPickle.load(open("data/test/reactions_full", "rb"))
    print(train_test_split(posts, 0.7, 0.1))
