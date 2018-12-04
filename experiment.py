import numpy as np
import cPickle
from matplotlib import pyplot as plt
import sys
import os

## models ##
# from PRET import PRET
from PRET_SVI import PRET_SVI
from evaluate import evaluate
from dataDUE_generator import dataDUELoader

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

# batch_rBp_dir = data_dir + "K10_batch_train/"
# batch_valid_on_shell_dir = data_dir + "K10_batch_on_shell/valid/"
# batch_valid_off_shell_dir = data_dir + "K10_batch_off_shell/valid/"
# batch_test_on_shell_dir = data_dir + "K10_batch_on_shell/test/"
# batch_test_off_shell_dir = data_dir + "K10_batch_off_shell/test/"
#
# meta_data_train_file = data_dir + "meta_data_train" + data_prefix
# meta_data_off_valid_file = data_dir + "meta_data_off_shell_valid" + data_prefix
# meta_data_off_test_file = data_dir + "meta_data_off_shell_test" + data_prefix


#### period_foxnews data ####
# data_dir = "data/period_foxnews/"
# data_prefix = "_period_foxnews_K10"
#
# batch_rBp_dir = data_dir + "period_foxnews_K10_batch_train/"
# batch_valid_on_shell_dir = data_dir + "period_foxnews_K10_batch_on_shell/valid/"
# batch_valid_off_shell_dir = data_dir + "period_foxnews_K10_batch_off_shell/valid/"
# batch_test_on_shell_dir = data_dir + "period_foxnews_K10_batch_on_shell/test/"
# batch_test_off_shell_dir = data_dir + "period_foxnews_K10_batch_off_shell/test/"
#
# meta_data_train_file = data_dir + "meta_data_train" + data_prefix
# meta_data_off_valid_file = data_dir + "meta_data_off_shell_valid" + data_prefix
# meta_data_off_test_file = data_dir + "meta_data_off_shell_test" + data_prefix
# meta_data_on_valid_file = data_dir + "meta_data_on_shell_valid" + data_prefix
# meta_data_on_test_file = data_dir + "meta_data_on_shell_test" + data_prefix


#### period_foxnews_nolike data ####
data_name = sys.argv[1]
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


def training(dataW, batch_rBp_dir, batch_valid_on_shell_dir=None, batch_valid_off_shell_dir=None, dataToken=None,
             dataDUE_loader=dataDUELoader, Model=PRET_SVI, hyperparameters = [], id_map_reverse = id_map_reverse, resume=None,
             batch_size=1024, lr_init=1.0, lr_kappa=0.1, random_shuffle=True,
             beta = 0.01, gamma=100, delta=0.01, zeta=0.1):
    K, G = hyperparameters
    model = Model(K, G)
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
    model._log("start training model %s, with hyperparameters %s"  % (str(Model.__name__), str(hyperparameters)))
    model._log("with data period_foxnews D: %d" % len(dataToken))
    if str(Model.__name__) == "PRET_SVI":
        model.log_file += "_K%d_G%d_batch_size_%d_lr_kappa_%f_beta_%f_gamma_%f_zeta_%f" % (K, G, batch_size, lr_kappa, beta, gamma, zeta)
        model.checkpoint_file += "_K%d_G%d_batch_size_%d_lr_kappa_%f_beta_%f_gamma_%f_zeta_%f" % (K, G, batch_size, lr_kappa, beta, gamma, zeta)
        model.fit(dataDUE, dataW, corpus=dataToken, resume= resume,
                  batch_size=batch_size, lr_init=lr_init, lr_kappa=lr_kappa,
                  beta=beta, gamma=gamma, delta=delta, zeta=zeta,
                  dataDUE_valid_on_shell=dataDUE_valid_on_shell, dataDUE_valid_off_shell=dataDUE_valid_off_shell)
    elif str(Model.__name__) == "PRET":
        model.log_file += "_K%d_G%d" % (K, G)
        model.fit(dataDUE, dataW, corpus=dataToken, resume=resume)


def testing(Model=PRET_SVI, hyperparameters=[], resume=None, dataDUE_loader=dataDUELoader,
            dataToken=None, id_map_reverse=id_map_reverse,
            batch_test_on_shell_dir=None, meta_data_on_test_file=None,
            batch_test_off_shell_dir=None, meta_data_off_test_file=None,):
    if batch_test_on_shell_dir is not None:
        dataDUE_test_on_shell = dataDUE_loader(meta_data_file=meta_data_on_test_file, batch_data_dir=batch_test_on_shell_dir, dataToken=dataToken, id_map=id_map_reverse,
                                                random_shuffle=False)
    else:
        dataDUE_test_on_shell = None
    if batch_test_off_shell_dir is not None:
        dataDUE_test_off_shell = dataDUE_loader(meta_data_file=meta_data_off_test_file, batch_data_dir=batch_test_off_shell_dir, dataToken=dataToken, id_map=id_map_reverse,
                                                 random_shuffle=False)
    else:
        dataDUE_test_off_shell = None

    K, G = hyperparameters
    model = Model(K, G)
    model._restoreCheckPoint(resume)

    result = performance(model=model, dataDUE_test_off_shell=dataDUE_test_off_shell, dataDUE_test_on_shell=dataDUE_test_on_shell)
    print result
    return result


def performance(model, dataDUE_test_off_shell=None, dataDUE_test_on_shell=None):
    result_on_shell = None
    result_off_shell = None
    if dataDUE_test_on_shell is not None:
        preds, trues = model.predict(dataDUE_test_on_shell, on_shell=True)
        result_on_shell = evaluate(preds, trues)
    if dataDUE_test_off_shell is not None:
        preds, trues = model.predict(dataDUE_test_off_shell, on_shell=False)
        result_off_shell = evaluate(preds, trues)
    return result_on_shell, result_off_shell


def modelDisplay(word_dictionary, Model=PRET_SVI, hyperparameters = [], resume=None):
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


if __name__ == "__main__":
    Model = eval(sys.argv[2])
    K = int(sys.argv[3])
    G = int(sys.argv[4])
    if sys.argv[5] == "full":
        batch_size = len(dataToken)
    else:
        batch_size = int(sys.argv[5])
    lr_kappa = float(sys.argv[6])
    beta = float(sys.argv[7])
    gamma = float(sys.argv[8])
    zeta = float(sys.argv[9])
    #K = 10
    #G = 6
    training(dataW, batch_rBp_dir,batch_valid_on_shell_dir=batch_valid_on_shell_dir, batch_valid_off_shell_dir=batch_valid_off_shell_dir,
             dataToken=dataToken,
             #Model=PRET,
             Model=Model,
             batch_size = batch_size, lr_init=1.0, lr_kappa=lr_kappa,
             beta=beta, gamma=gamma, zeta=zeta,
             hyperparameters=[K, G],
             id_map_reverse = id_map_reverse,
             resume=None)
#             resume = "ckpt/CNN/PRET_SVI_K50_G16_batch_size_23000_lr_kappa_0.000000_beta_%f_gamma_%f_zeta_0.100000" % (beta, gamma))

    # testing(Model=PRET_SVI, hyperparameters=[K, G],
    #         resume="ckpt/period_foxnews_nolike/PRET_SVI_K50_G16_batch_size_23000_lr_kappa_0.000000_beta_0.010000_gamma_100.000000_zeta_0.100000_epoch_017",
    #         dataToken=dataToken, id_map_reverse=id_map_reverse,
    #         batch_test_on_shell_dir=batch_test_on_shell_dir, meta_data_on_test_file=meta_data_on_test_file,
    #         batch_test_off_shell_dir=batch_test_off_shell_dir, meta_data_off_test_file=meta_data_off_test_file)

