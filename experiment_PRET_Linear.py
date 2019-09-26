import numpy as np
import cPickle
from matplotlib import pyplot as plt
import sys
import os
import argparse
import re

## models ##
# from PRET import PRET
# from PRET_SVI import PRET_SVI
from PRET_Linear_SVI import PRET_Linear_SVI
from evaluate import evaluate
from dataDUE_generator_full_read import dataDUELoader

EMOTICON_LIST = ["LIKE", "LOVE", "SAD", "WOW", "HAHA", "ANGRY"]


data_dir = "data/CNN_foxnews/"
data_prefix = "_CNN_foxnews_combined_K10"

id_map_file = data_dir + "id_map" + data_prefix
postcontent_dataW_file = data_dir + "dataW" + data_prefix
postcontent_dataToken_file = data_dir + "dataToken" + data_prefix
word_dictionary_file = data_dir + "word_dictionary" + data_prefix

id_map, id_map_reverse = cPickle.load(open(id_map_file, "r"))
dataW = cPickle.load(open(postcontent_dataW_file, "r"))
dataToken = cPickle.load(open(postcontent_dataToken_file, "r"))
word_dictionary = cPickle.load(open(word_dictionary_file, "r"))


def training(
        dataW,
        batch_rBp_dir,
        batch_valid_on_shell_dir=None,
        batch_valid_off_shell_dir=None,
        dataToken=None,
        dataDUE_loader=dataDUELoader,
        Model=PRET_Linear_SVI,
        hyperparameters=[],
        id_map_reverse=id_map_reverse,
        resume=None,
        batch_size=1024,
        lr_init=1.0,
        lr_kappa=0.1,
        random_shuffle=True,
        beta=0.01,
        gamma=100.0,
        gamma_z=1.0,
        gamma_u=0.1,
        delta=0.01,
        zeta=0.1,
        omega=1.0
):
    K1, K2, G = hyperparameters
    model = Model(K1, K2, G)
    dataDUE = dataDUE_loader(meta_data_file=meta_data_train_file, batch_data_dir=batch_rBp_dir, dataToken=dataToken, id_map=id_map_reverse,
                             random_shuffle=random_shuffle)
    if batch_valid_on_shell_dir is not None:
        dataDUE_valid_on_shell = dataDUE_loader(meta_data_file=meta_data_on_valid_file, batch_data_dir=batch_valid_on_shell_dir, dataToken=dataToken, id_map=id_map_reverse,
                                                random_shuffle=False)
    else:
        dataDUE_valid_on_shell = None
    if batch_valid_off_shell_dir is not None:
        dataDUE_valid_off_shell = dataDUE_loader(meta_data_file=meta_data_off_valid_file, batch_data_dir=batch_valid_off_shell_dir, dataToken=dataToken, id_map=id_map_reverse,
                                                 random_shuffle=False)
    else:
        dataDUE_valid_off_shell = None

    model.log_file = log_dir + Model.__name__
    model.checkpoint_file = ckpt_dir + Model.__name__
    model._log("start training model %s, with hyperparameters %s" % (str(Model.__name__), str(hyperparameters)))
    model._log("with data period_foxnews D: %d" % len(dataToken))

    str_pars = "_K1_%d_K2_%d_G%d_batch_size_%d_lr_kappa_%s_beta_%s_gamma_%s_gz_%s_gu_%s_delta_%s_zeta_%s_omega_%s" % \
        (
          K1, K2, G, batch_size,
          str(lr_kappa), str(beta), str(gamma), str(gamma_z), str(gamma_u),
          str(delta), str(zeta), str(omega)
        )
    str_pars = re.sub(' ', ',', str_pars)
    model.log_file += str_pars
    model.checkpoint_file += str_pars
    model.fit(dataDUE, dataW, corpus=dataToken, resume= resume,
              batch_size=batch_size, lr_init=lr_init, lr_kappa=lr_kappa,
              beta=beta, gamma=gamma, gamma_z=gamma_z, gamma_u=gamma_u, delta=delta, zeta=zeta, omega=omega,
              dataDUE_valid_on_shell=dataDUE_valid_on_shell, dataDUE_valid_off_shell=dataDUE_valid_off_shell)


def testing(Model=PRET_Linear_SVI, hyperparameters=[], resume=None, dataDUE_loader=dataDUELoader,
            dataToken=None, id_map_reverse=id_map_reverse,
            batch_test_on_shell_dir=None, meta_data_on_test_file=None,
            batch_test_off_shell_dir=None, meta_data_off_test_file=None, ec_sets=None):

    K1, K2, G = hyperparameters
    model = Model(K1, K2, G)
    model._restoreCheckPoint(resume)

    results = []

    for id_map_reverse_filtered in id_map_filter_ec_sets(id_map=id_map_reverse, ec_sets=ec_sets):
        if batch_test_on_shell_dir is not None:
            dataDUE_test_on_shell = dataDUE_loader(meta_data_file=meta_data_on_test_file, batch_data_dir=batch_test_on_shell_dir, dataToken=dataToken, id_map=id_map_reverse_filtered,
                                                    random_shuffle=False)
        else:
            dataDUE_test_on_shell = None
        if batch_test_off_shell_dir is not None:
            dataDUE_test_off_shell = dataDUE_loader(meta_data_file=meta_data_off_test_file, batch_data_dir=batch_test_off_shell_dir, dataToken=dataToken, id_map=id_map_reverse,
                                                     random_shuffle=False)
        else:
            dataDUE_test_off_shell = None

        result = performance(model=model, dataDUE_test_off_shell=dataDUE_test_off_shell, dataDUE_test_on_shell=dataDUE_test_on_shell)
        print result
        results.append(result)
    return results


def id_map_filter_ec_sets(id_map, ec_sets):
    """
    :param id_map: from post_id to doc_id
    :param ec_sets:
    :return:
    """
    if ec_sets is None:
        return [id_map]
    current_ec_set = set()
    id_maps = []
    for i in range(len(ec_sets)):
        ec_set = set([ec[0] for ec in ec_sets[i]])
        current_ec_set = current_ec_set.union(ec_set)
        id_map_new = dict()
        for id in id_map:
            if id_map[id] in current_ec_set:
                id_map_new[id] = id_map[id]
        id_maps.append(id_map_new)
    return id_maps


def performance(model, dataDUE_test_off_shell=None, dataDUE_test_on_shell=None):
    result_on_shell = None
    result_off_shell = None
    if dataDUE_test_on_shell is not None:
        preds, trues = model.predict(dataDUE_test_on_shell, on_shell=True)
        if len(trues) == 0:
            result_on_shell = [None, None, None, None]
        else:
            result_on_shell = evaluate(preds, trues)
    if dataDUE_test_off_shell is not None:
        preds, trues = model.predict(dataDUE_test_off_shell, on_shell=False)
        result_off_shell = evaluate(preds, trues)
    return result_on_shell, result_off_shell


def modelDisplay(word_dictionary, Model=PRET_Linear_SVI, hyperparameters = [], resume=None):
    K1, K2, G = hyperparameters
    model = Model(K1, K2, G)
    model._restoreCheckPoint(resume)

    K = K2

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
    elif Model.__name__ == "PRET_Linear_SVI":
        theta = model.GLV["theta2"]
        pi = model.GLV["pi"]
        eta = model.GLV["eta"]
        psi = model.GLV["psi"]
        phiB = np.sum(model.GLV["phi1"], axis=0)
        phiT = model.GLV["phi2"]

    # find top words for each topic #
    n_top_words = 8
    for i, topic_dist in enumerate(phiT.tolist()):
        topic_words = np.array(word_dictionary)[np.argsort(topic_dist)][:-n_top_words:-1]
        print "Topic {}: {}".format(i, ','.join(topic_words))
        # print topic_dist
    n_top_words_B = 2 * n_top_words
    topic_words = np.array(word_dictionary)[np.argsort(phiB)][:-n_top_words_B:-1]
    print "Topic {}: {}".format("B", ",".join(topic_words))

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # corpus-level topic distribution #
    ax1.plot(theta)
    ax1.set_title("corpus-level topic distribution")
    # ax1.legend()
    # topic-background distribution #
    labels_B = ["Background", "topic"]
    ax2.pie(pi, labels=labels_B, autopct='%1.1f%%')
    ax2.set_title("topic-background distribution")
    # user-group distribution cumulative #
    ax3.plot(np.mean(psi, axis=0))
    ax3.set_title("mean user-group distribution")
    plt.legend()
    plt.show()

    f, axes = plt.subplots(1, G)
    # topic-emotion distribution for each group #
    for g in range(G):
        for k in range(K):
            axes[g].plot(eta[k, g, :], label="topic %d" % k)
        axes[g].set_title("topic-emotion distribution for group %d" % g)
    plt.legend()
    plt.show()

    # z = model.z
    # for d in range(z.shape[0]):
    #     print "doc %d:" %d
    #     document = [word_dictionary[i] for i in dataToken[d]]
    #     print document
    #     print "post_id", id_map[d]
    #     print "topic", z[d]


class ArgParse(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-dm", "--data_name", default="CNN_nolike")
        parser.add_argument("-K1", "--K1", default=5, type=int)
        parser.add_argument("-K2", "--K2", default=7, type=int)
        parser.add_argument("-G", "--G", default=5, type=int)
        parser.add_argument("-bs", "--batch_size", default=23000, type=int)
        parser.add_argument("-kappa", "--kappa", default=0.0, type=float)
        parser.add_argument("-beta", "--beta", default=0.01, type=float)
        parser.add_argument("-gamma", "--gamma", default=1.0, type=float)
        parser.add_argument("-gamma_z", "--gamma_z", default=1.0, type=float)
        parser.add_argument("-gamma_u", "--gamma_u", default=0.1, type=float)
        parser.add_argument("-delta", "--delta", default=1.0, type=float)
        parser.add_argument("-delta1", "--delta1", default=1.0, type=float)
        parser.add_argument("-zeta", "--zeta", default=0.1, type=float)
        parser.add_argument("-omega", "--omega", default=1.0, type=float)
        parser.add_argument("-omega1", "--omega1", default=1.0, type=float)
        parser.add_argument("-omega2", "--omega2", default=10.0, type=float)
        self.parser = parser

    def parse_args(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    args = ArgParse().parse_args()

    data_name = args.data_name
    data_dir = "data/" + data_name + "/"

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

    log_dir = "log/" + data_name + "/"
    ckpt_dir = "ckpt/" + data_name + "/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    training(
        dataW=dataW,
        batch_rBp_dir=batch_rBp_dir,
        batch_valid_on_shell_dir=batch_valid_on_shell_dir,
        batch_valid_off_shell_dir=batch_valid_off_shell_dir,
        dataToken=dataToken,
        Model=PRET_Linear_SVI,
        batch_size=args.batch_size,
        lr_init=1.0,
        lr_kappa=args.kappa,
        beta=args.beta,
        gamma=args.gamma,
        gamma_z=args.gamma_z,
        gamma_u=args.gamma_u,
        delta=args.delta if args.delta1 is None else np.array([args.delta, args.delta1]),
        zeta=args.zeta,
        omega=args.omega if args.omega1 is None else np.array([args.omega, args.omega1, args.omega2]),
        hyperparameters=[args.K1, args.K2, args.G],
        id_map_reverse=id_map_reverse,
        resume=None
    )
#       resume = "ckpt/CNN/PRET_SVI_K50_G16_batch_size_23000_lr_kappa_0.000000_beta_%f_gamma_%f_zeta_0.100000" % (beta, gamma))

    # ec_sets = cPickle.load(open("result/%s_train_posts_divide" % data_name, 'rb'))
    # print(list(map(lambda x: len(x), id_map_filter_ec_sets(id_map=id_map_reverse, ec_sets=ec_sets))))
    # results = testing(Model=PRET_SVI, hyperparameters=[K, G],
    #         resume="ckpt/CNN_nolike/PRET_SVI_K50_G16_batch_size_23000_lr_kappa_0.000000_beta_1.000000_gamma_1.000000_zeta_0.100000_best_[2]",
    #         dataToken=dataToken, id_map_reverse=id_map_reverse,
    #         batch_test_on_shell_dir=batch_test_on_shell_dir, meta_data_on_test_file=meta_data_on_test_file,
    #         batch_test_off_shell_dir=batch_test_off_shell_dir, meta_data_off_test_file=meta_data_off_test_file,
    #         ec_sets=ec_sets)
    # print(list(map(lambda x: x[0][2], results)))


