import cPickle
import sys
import numpy as np

from dataDUE_generator import dataDUELoader


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

# data_name = sys.argv[1]
data_name = "period_foxnews_nolike"
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

if __name__ == "__main__":
    dataDUE_train = dataDUELoader(
        meta_data_file=meta_data_train_file,
        batch_data_dir=batch_rBp_dir,
        dataToken=dataToken,
        id_map=id_map_reverse,
        random_shuffle=False
    )

    emoticon_count = dict()
    for docdata in dataDUE_train.generate():
        d, docToken, [doc_u, doc_e] = docdata
        emoticon_count[d] = len(doc_e)

    with open("result/" + data_name + "_train_emoticon_count", 'wb') as f:
        cPickle.dump(emoticon_count, f)

    ec = []
    for doc_id in emoticon_count:
        ec.append((doc_id, emoticon_count[doc_id]))
    ec.sort(key=lambda x: x[1])
    d = 10
    b = len(ec) // d
    a = 0
    ec_sets = []
    while a < len(ec) - 1:
        print("%d-th posts: emoticons %d" % (a, ec[a][1]))
        a = a + b
        if a + b > len(ec):
            ec_sets.append(set(ec[a-b:a+b]))
            break
        else:
            ec_sets.append(set(ec[a-b:a]))
        print("last_set size: %d" % len(ec_sets[-1]))
    with open("result/" + data_name + "_train_posts_divide", "wb") as f:
        cPickle.dump(ec_sets, f)

    print("%d-th posts: emoticons %d" % (len(ec) - 1, ec[-1][1]))
    hist = np.histogram(ec)
    print(hist)
