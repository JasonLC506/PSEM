import numpy as np
import re
import cPickle
from matplotlib import pyplot as plt
import seaborn as sns

## models ##
from PRET import PRET
from PRET_SVI import PRET_SVI

pattern_CNN = re.compile(r'^5550296508_')
pattern_foxnews = re.compile(r'^15704546335_')

EMOTICON_LIST = ["LOVE", "SAD", "WOW", "HAHA", "ANGRY"]

def modelDisplay(word_dictionary, id_map, posts_content_raw, user_dist_two_file,
                 Model=PRET_SVI, hyperparameters = [], resume=None):
    K, G = hyperparameters
    model = Model(K, G)
    model._restoreCheckPoint(resume)

    ## extract paras #
    if Model.__name__ == "PRET":
        theta = model.theta
        pi = model.pi
        eta = model.eta
        psi = model.psi
        phiB = model.phiB
        phiT = model.phiT
    elif Model.__name__ == "PRET_SVI":
        theta = model.GLV["theta"]
        pi = model.GLV["pi"]
        eta = model.GLV["eta"]
        psi = model.GLV["psi"]
        phiB = model.GLV["phiB"]
        phiT = model.GLV["phiT"]
        z = model.z

    with open(user_dist_two_file, "r") as f:
        user_dist_two = cPickle.load(f)
    user_list_CNN = []
    user_list_foxnews = []
    user_list_both = []
    for uid in user_dist_two:
        ur = user_dist_two[uid]
        if ur[0]>0 and ur[1]>0:
            user_list_both.append(uid)
        elif ur[0]>0:
            user_list_CNN.append(uid)
        elif ur[1]>0:
            user_list_foxnews.append(uid)
        else:
            raise ValueError("user with no reactions %d" % uid)

    psi_mean_CNN = np.mean(psi[np.array(user_list_CNN), :], axis=0)
    psi_mean_foxnews = np.mean(psi[np.array(user_list_foxnews), :], axis=0)
    psi_mean_both = np.mean(psi[np.array(user_list_both), :], axis=0)

    print "psi_mean_CNN", psi_mean_CNN
    print "psi_mean_foxnews", psi_mean_foxnews
    print "psi_mean_both", psi_mean_both
    print "psi all", np.mean(psi, axis=0)


    # find top words for each topic #
    n_top_words = 8
    n_top_docs = 3
    for k, topic_dist in enumerate(phiT.tolist()):
        topic_words = np.array(word_dictionary)[np.argsort(topic_dist)][:-n_top_words:-1]
        # topic_words_weights = np.array(topic_dist)[np.argsort(topic_dist)][:-n_top_words:-1]
        print "Topic {}: {}".format(k, ','.join(topic_words))
        # print "    weights: %s" % ",".join(map(lambda x: "%.2g" % x, topic_words_weights))

        # topic_doc_ds = np.argsort(z[:,k])[:-n_top_docs-1:-1]
        # topic_docs = []
        # topic_docs_flag = []
        # topic_docs_ids = np.array(id_map)[topic_doc_ds]
        # print "################# top topic documents ##################"
        # for i in range(topic_doc_ds.shape[0]):
        #     topic_docs.append(posts_content_raw[topic_docs_ids[i]])
        #     if pattern_CNN.match(topic_docs_ids[i]) is not None:
        #         topic_docs_flag.append("CNN")
        #     elif pattern_foxnews.match(topic_docs_ids[i]) is not None:
        #         topic_docs_flag.append("foxnews")
        #     else:
        #         raise ValueError(topic_docs_ids)
        #     print topic_docs[i] + "  -- " + topic_docs_flag[i]

        print ""

        # print topic_dist
    # n_top_words_B = 2 * n_top_words
    # topic_words = np.array(word_dictionary)[np.argsort(phiB)][:-n_top_words_B:-1]
    # print "Topic {}: {}".format("B", ",".join(topic_words))

    f, axes = plt.subplots(2,3, sharex=True, sharey=True)
    f.delaxes(axes.flatten()[-1])
    cbar_ax = f.add_axes([.8, .1, .03, .3])
    for g in range(G):
        r, c = g/3, g%3
        axis = axes[r,c]
        sns.heatmap(eta[:,g,:], ax=axis, cmap="Blues",
                    cbar=g == 0,
                    vmin=0, vmax=1,
                    cbar_ax=None if g else cbar_ax)
        # cax = axis.heatmap(eta[k,:,:], cmap='Blues')
        axis.set_xticks(np.arange(len(EMOTICON_LIST))+0.5)
        axis.set_xticklabels(map(lambda x: x[0], EMOTICON_LIST))
        # axis.set_xlabel("group")
        # axis.set_ylabel("emotion")
        axis.set_title("group %d" % g)
    plt.legend()
    plt.show()

    #
    # f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # # corpus-level topic distribution #
    # ax1.plot(theta)
    # ax1.set_title("corpus-level topic distribution")
    # # ax1.legend()
    # # topic-background distribution #
    # labels_B = ["Background", "topic"]
    # ax2.pie(pi, labels=labels_B, autopct='%1.1f%%')
    # ax2.set_title("topic-background distribution")
    # # user-group distribution cumulative #
    # ax3.plot(np.mean(psi, axis=0))
    # ax3.set_title("mean user-group distribution")
    # plt.legend()
    # plt.show()
    #
    # f, axes = plt.subplots(1, G)
    # # topic-emotion distribution for each group #
    # for g in range(G):
    #     for k in range(K):
    #         axes[g].plot(eta[k, g, :], label="topic %d" % k)
    #     axes[g].set_title("topic-emotion distribution for group %d" % g)
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    data_dir = "data/CNN_foxnews/"
    data_prefix = "_CNN_foxnews_combined_K10"

    id_map_file = data_dir + "id_map" + data_prefix
    postcontent_dataW_file = data_dir + "dataW" + data_prefix
    postcontent_dataToken_file = data_dir + "dataToken" + data_prefix
    postcontent_raw_file = data_dir + "posts_content_raw"
    word_dictionary_file = data_dir + "word_dictionary" + data_prefix

    id_map, id_map_reverse = cPickle.load(open(id_map_file, "r"))
    dataW = cPickle.load(open(postcontent_dataW_file, "r"))
    dataToken = cPickle.load(open(postcontent_dataToken_file, "r"))
    word_dictionary = cPickle.load(open(word_dictionary_file, "r"))
    posts_content_raw = cPickle.load(open(postcontent_raw_file, "r"))

    data_dir = "data/period_foxnews_nolike/"

    batch_rBp_dir = data_dir + "train/"
    batch_valid_on_shell_dir = data_dir + "on_shell/valid/"
    batch_valid_off_shell_dir = data_dir + "off_shell/valid/"
    batch_test_on_shell_dir = data_dir + "on_shell/test/"
    batch_test_off_shell_dir = data_dir + "off_shell/test/"

    meta_data_train_file = data_dir + "meta_data_train"
    meta_data_off_valid_file = data_dir + "meta_data_off_shell_valid"
    meta_data_off_test_file = data_dir + "meta_data_off_shell_test"
    meta_data_on_valid_file = data_dir + "meta_data_on_shell_valid"
    meta_data_on_test_file = data_dir + "meta_data_on_shell_test"

    user_dist_two_file = data_dir + "user_dist_TWO_K10"

    K = 7
    G = 5

    modelDisplay(word_dictionary=word_dictionary, id_map=id_map, posts_content_raw=posts_content_raw, user_dist_two_file=user_dist_two_file,
                 Model=PRET_SVI, hyperparameters=[K,G],
                 resume="ckpt_test/period_foxnews_nolike/PRET_SVI_K7_G5_batch_size_23000_lr_kappa_0.000000_beta_0.010000_gamma_100.000000_zeta_0.100000_best_ppl[4]")