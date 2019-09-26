import numpy as np
# from functions import multinomial, probNormalize
import cPickle
import os
from datetime import datetime
import random
import math


class dataDUELoader(object):
    def __init__(self, meta_data_file, batch_data_dir, id_map, dataToken=None, max_qsize = 5000, random_shuffle=False):

        self.E = 0                                  # dimension of emotion
        self.U = 0                                  # number of users
        self.Md = None                              # count of document-level total emoticons List[D]
        self.Nd = None                              # count of document-level total tokens List[D]
        self.D = 0                                  # number of documents
        self.D_current_data = 0                     # number of documents in current dataset batch_data_dir
        self.data_dir = batch_data_dir                    # data_directory
        self.V = 0
        with open(meta_data_file, "r") as f:
            meta_data = cPickle.load(f)
            self.E = meta_data["E"]
            self.U = meta_data["U"]
            self.D = meta_data["D"]
            self.Md = meta_data["Md"]
            self.Nd = meta_data["Nd"]
            self.V = meta_data["V"]
            self.D_current_data = min(meta_data["D_current_data"], len(id_map))
            ## with CNN_only data ##
            # self.Md = [0 for i in range(self.D)]
            # for pid in meta_data["Md"]:
            #     if pid not in id_map:
            #         # print pid    ### test
            #         continue
            #
            #     self.Md[id_map[pid]] = meta_data["Md"][pid]


        self.id_map = id_map                        # id_map between post_id and document_id
        self.batch_data_dir = batch_data_dir

        self.D_train = len(self.Md)

        self.dataToken = dataToken

        # for data generation, ! not implemented yet ! #
        # will modify self._dataBatchReader() #
        # self.batch_size = batch_size
        self.random_shuffle = random_shuffle

        # multiprocess data reader #
        self.data = []
        self._dataFullRead()

    def _dataFullRead(self):
        file_list = os.listdir(self.batch_data_dir)
        cnt = 0
        for fn in file_list:
            start = datetime.now()
            with open(os.path.join(self.batch_data_dir, fn), "r") as f:
                posts = cPickle.load(f)
            # duration = datetime.now() - start
            # print "_dataBatchReader: load %s takes %f s" % (fn, duration.total_seconds())
            post_id_list = posts.keys()
            if self.random_shuffle:
                random.shuffle(post_id_list)
            for post_id in post_id_list:
                if post_id not in self.id_map:
                    continue
                document_id = self.id_map[post_id]
                if self.dataToken is not None:
                    self.data.append([
                        document_id,
                        np.array(self.dataToken[document_id], dtype=np.int64),
                        map(lambda x: np.array(x, dtype=np.int64), posts[post_id])
                    ])  # set max waiting time
                else:
                    self.data.append([
                        document_id,
                        posts[post_id]
                    ])   # set max waiting time
                cnt += 1
            del posts
        self.D_current_data = cnt

    def batchGenerate(self, batch_size=1, keep_complete=True):
        data_size = len(self.data)
        index = np.arange(data_size)
        if self.random_shuffle:
            np.random.shuffle(index)
        if batch_size > data_size / 2:
            # warnings.warn("too large batch_size %d compared with data_size %d, set to data_size" %
            #               (batch_size, self.data_size))
            batch_size = data_size
        N_batch = int(math.ceil(float(data_size) / float(batch_size)))
        # incomplete_batch = False
        # if self.D_current_data % batch_size != 0:
        #     N_batch += 1
        #     incomplete_batch = True
        for i_batch in xrange(N_batch):
            batch_index = index[i_batch * batch_size: min(data_size, (i_batch + 1) * batch_size)]
            if keep_complete:
                batch_size_temp = batch_index.shape[0]
                if batch_size_temp < batch_size:
                    batch_index = np.concatenate(
                        [batch_index, index[:(batch_size - batch_size_temp)]],
                        axis=0
                    )
            batch_size_real = batch_index.shape[0]
            yield i_batch, batch_size_real, self.generateSingleBatch(batch_index)

    def generateSingleBatch(self, batch_index):
        indices = batch_index.tolist()
        for index in indices:
            yield self.data[index]

    def generate(self, timeout=100):
        # depends on inputs, yield [document_id, [[reader_id],[emoticon]]] #
        ### example ###
        # for d in range(self.D):
        #     yield d, [np.arange(self.Md[d]), multinomial(probNormalize(np.random.random(self.E)), self.Md[d])]
        ###############
        """
        iteratively generate data for one epoch
        :param timeout: patience for waiting data
        """
        for doc_cnt in range(self.D_current_data):
            yield self.data[doc_cnt]


if __name__ == "__main__":
    data_prefix = "CNN_K10_"
    batch_rBp_dir = "data/" + data_prefix + "reactionsByPost_batch"
    meta_data_file = "data/" + data_prefix + "meta_data"
    id_map_file = "data/" + data_prefix + "post_id_map"

    id_map, id_map_reverse = cPickle.load(open(id_map_file, "r"))

    dataDUE = dataDUELoader(meta_data_file=meta_data_file, batch_data_dir=batch_rBp_dir, id_map=id_map_reverse)



