"""
PRET with Linear term added
"""
import numpy as np
from datetime import datetime
from tqdm import tqdm
import cPickle
import warnings

from functions import EDirLog, probNormalize, probNormalizeLog, expConstantIgnore
from evaluate import evaluate
from PRET_SVI import latentVariableGlobal


np.seterr(divide='raise', over='raise')
tqdm.monitor_interval = 0


class PRET_Linear_SVI(object):
    def __init__(self, K1, K2, G):
        # model hyperparameters #
        self.alpha2 = 1.0 / float(K2)
        self.alpha1 = 1.0 / float(K1)
        self.beta = 0.01
        self.gamma = 10000
        self.gamma_z = self.gamma_u = self.gamma
        self.delta = 0.001
        self.zeta = 0.01
        self.omega = 1.0

        # data dimension #
        self.E = 0                                              # number of emotions
        self.K2 = K2
        self.K1 = K1
        self.G = G                                              # number of groups
        self.D = 0                                              # number of documents
        self.D_train = 0                                        # number of documents in training set
        self.Nd = []                                            # number of words of documents (varying over docs)
        self.Md = []                                            # number of emotions of documents (varying over docs)
        self.V = 0                                              # size of vocabulary
        self.U = 0                                              # number of users

        # global #
        self.theta2 = latentVariableGlobal()
        self.theta1 = latentVariableGlobal()
        self.pi = latentVariableGlobal()
        self.phi2 = latentVariableGlobal()
        self.phi1 = latentVariableGlobal()
        self.eta = latentVariableGlobal()
        self.eta_z = latentVariableGlobal()
        self.eta_u = latentVariableGlobal()
        self.psi = latentVariableGlobal()
        self.se = latentVariableGlobal()

        # local #
        self.z2 = None
        self.z1 = None
        self.y = None
        self.x = None
        self.r = None

        # model global latent variables point estimate #
        self.GLV = {
            "theta2": None, "theta1": None, "pi": None, "phi2": None, "phi1": None, "eta": None,
            "eta_z": None, "eta_u": None, "psi": None, "se": None
        }

        # stochastic learning #
        self.lr = None                                          # learning rate pars
        self.converge_threshold_inner = None

        # save & store #
        self.checkpoint_file = "ckpt/PRET_Linear_SVI"
        self.epoch_init = 0
        self.log_file = "log/PRET_Linear_SVI"

    def fit(
            self, dataDUE, dataW, dataDUE_valid_on_shell=None, dataDUE_valid_off_shell=None, corpus=None,
            alpha2=0.1, alpha1=0.1, beta=0.01, gamma=10000.0, gamma_z=1.0, gamma_u=1.0, delta=0.001, zeta=0.01, omega=1.0,
            max_iter=500, resume=None,
            batch_size=1024, N_workers=4, lr_tau=2, lr_kappa=0.1, lr_init=1.0, converge_threshold_inner=0.001
    ):
        """
        stochastic variational inference
        :param dataDUE: data generator for each document id, generate [[reader_id], [emoticon]]
        :param dataW: Indexed corpus                    np.ndarray([self.D, self.V]) scipy.sparse.csr_matrix
        """
        dataToken = corpus               # ! ensure corpus/dataToken is input

        self._setDataDimension(dataDUE=dataDUE, dataW=dataW, dataToken=dataToken)

        self._setHyperparameters(
            alpha2=alpha2, alpha1=alpha1, beta=beta, gamma=gamma, gamma_z=gamma_z, gamma_u=gamma_u, delta=delta,
            zeta=zeta, omega=omega
        )

        self.lr = {"tau": lr_tau, "kappa": lr_kappa, "init": lr_init}
        self.converge_threshold_inner = converge_threshold_inner            # inner iteration for each document

        if batch_size > self.D_train/2:
            batch_size = self.D_train
            print "set batch_size full %d" % self.D_train

        if resume is None:
            self._initialize(dataDUE=dataDUE, dataW=dataW, dataToken=dataToken)
        else:
            self._restoreCheckPoint(filename=resume)

        self._estimateGlobal()
        ppl_initial, perf_initial = self.validate(
            dataDUE_valid_off_shell=dataDUE_valid_off_shell,
            dataDUE_valid_on_shell=dataDUE_valid_on_shell
        )
        self._log("before training, ppl: %s\nperf: %s\n" % (str(ppl_initial), str(perf_initial)))
        perf_best = perf_initial

        if batch_size >= self.D_train:
            batch_size = self.D_train
            self._log("full batch, batch_size set to D_train %d" % self.D_train)

        for epoch in range(self.epoch_init, max_iter):
            self._fit_single_epoch(
                dataDUE=dataDUE, dataW=dataW, dataToken=dataToken, epoch=epoch, batch_size=batch_size
            )
            self._estimateGlobal()
            ppl, perf = self.validate(
                dataDUE_valid_off_shell=dataDUE_valid_off_shell,
                dataDUE_valid_on_shell=dataDUE_valid_on_shell,
                epoch=epoch
            )
            print("epoch: %d, ppl: %s\nperf: %s\n" % (epoch, str(ppl), str(perf)))
            ### test ###
            # ppl_off_shell_for_on_shell = self._ppl(dataDUE_valid_on_shell, epoch, on_shell=False)
            # print "ppl_off_shell for on_shell", str(ppl_off_shell_for_on_shell)

            ppl_best, best_flag = self._ppl_compare(perf_best, perf)
            self._log("epoch: %d, ppl: %s\nperf: %s\n" % (epoch, str(ppl), str(perf)))
            self._saveCheckPoint(epoch, ppl=ppl, perf=perf)
            for i in range(len(best_flag)):
                if i not in [0, 2, 4, 6]:             # auc-pr micro
                    continue
                if best_flag[i]:
                    self._saveCheckPoint(
                        epoch=epoch,
                        ppl=ppl,
                        perf=perf,
                        filename=self.checkpoint_file + "_best_[%d]" % i
                    )

    def validate(self, dataDUE_valid_on_shell, dataDUE_valid_off_shell, epoch=-1):
        if dataDUE_valid_on_shell is None:
            ppl_on_shell = [None, None, None]
            perf_on_shell = [None]
        else:
            ppl_on_shell = self._ppl(dataDUE_valid_on_shell, epoch=epoch, on_shell=True)
            preds, trues = self.predict(dataDUE_valid_on_shell, on_shell=True)
            perf_on_shell = evaluate(preds=preds, trues=trues)

        if dataDUE_valid_off_shell is None:
            ppl_off_shell = [None, None, None]
            perf_off_shell = [None]
        else:
            ppl_off_shell = self._ppl(dataDUE_valid_off_shell, epoch=epoch, on_shell=False)
            preds, trues = self.predict(dataDUE_valid_off_shell, on_shell=False)
            perf_off_shell = evaluate(preds=preds, trues=trues)
        return ppl_on_shell + ppl_off_shell, perf_on_shell + perf_off_shell

    def predict(self, dataDUE, on_shell=False):
        start = datetime.now()
        self._estimateGlobal()
        preds = []
        trues = []
        for docdata in dataDUE.generate():
            if on_shell:
                result = self._predict_single_document_on_shell(docdata)
            else:
                result = self._predict_single_document_off_shell(docdata)
            preds += result[0]
            trues += result[1]
        duration = (datetime.now() - start).total_seconds()
        print "predict takes %fs" % duration
        return np.array(preds), np.array(trues)

    def _setDataDimension(self, dataDUE, dataW, dataToken):
        self.E = dataDUE.E
        self.U = dataDUE.U
        self.Md = dataDUE.Md
        self.D = dataDUE.D
        self.D_train = dataDUE.D_current_data
        self.Nd = map(lambda x: len(x), dataToken)
        self.V = dataDUE.V
        print "set data dimension,", "D", self.D, "D_train", self.D_train

    def _setHyperparameters(self, alpha2, alpha1, beta, gamma, gamma_z, gamma_u, delta, zeta, omega):
        self.alpha2 = 1.0 / float(self.K2)        # fixed based on [2]
        self.alpha1 = 1.0 / float(self.K1)
        self.beta = beta
        # self.gamma = beta * self.V * sum(self.Md) / (1.0 * self.E * sum(self.Nd) * self.G)
        self.gamma = gamma
        self.gamma_z = gamma_z
        self.gamma_u = gamma_u
        self.delta = delta
        self.zeta = zeta
        self.omega = omega
        self._log(
            "set up hyperparameters: " +
            "alpha2=%s, alpha1=%s, beta=%s, gamma=%s, gamma_z=%s, gamma_u=%s, delta=%s, zeta=%s, omega=%s" %
            (
                str(self.alpha2), str(self.alpha1), str(self.beta), str(self.gamma), str(self.gamma_z),
                str(self.gamma_u), str(self.delta), str(self.zeta), str(self.omega)
            )
        )

    def _initialize(self, dataDUE, dataW, dataToken):
        start = datetime.now()
        print "start _initialize"

        self.z2 = probNormalize(np.random.random([self.D, self.K2]))
        self.z1 = probNormalize(np.random.random([self.D, self.K1]))
        # self.y = []
        # self.x = []
        # for d in range(self.D):
        #     self.y.append(probNormalize(np.random.random([self.Nd[d], 2])))
        #     self.x.append(probNormalize(np.random.random([self.Md[d], self.G])))

        self.theta2.initialize(shape=[self.K2], seed=self.alpha2)
        self.theta1.initialize(shape=[self.K1], seed=self.alpha1)
        self.pi.initialize(shape=[2], seed=self.delta)
        self.phi1.initialize(shape=[self.K1, self.V], seed=self.beta)
        self.phi2.initialize(shape=[self.K2, self.V], seed=self.beta)
        self.eta.initialize(shape=[self.K2, self.G, self.E], seed=self.gamma, additional_noise_axis=1)
        self.eta_z.initialize(shape=[self.K1, self.E], seed=self.gamma_z)
        self.eta_u.initialize(shape=[self.U, self.E], seed=self.gamma_u)
        self.psi.initialize(shape=[self.U, self.G], seed=self.zeta)
        self.se.initialize(shape=[3], seed=self.omega)

        duration = (datetime.now() - start).total_seconds()
        print "_initialize takes %fs" % duration

    def _estimateGlobal(self):
        """
        give point estimate of global latent variables, self.GLV
        current: mean
        """
        self.GLV["theta2"] = probNormalize(self.theta2.data)
        self.GLV["theta1"] = probNormalize(self.theta1.data)
        self.GLV["pi"] = probNormalize(self.pi.data)
        self.GLV["phi1"] = probNormalize(self.phi1.data)
        self.GLV["phi2"] = probNormalize(self.phi2.data)
        self.GLV["eta"] = probNormalize(self.eta.data)
        self.GLV["eta_z"] = probNormalize(self.eta_z.data)
        self.GLV["eta_u"] = probNormalize(self.eta_u.data)
        self.GLV["psi"] = probNormalize(self.psi.data)
        self.GLV["se"] = probNormalize(self.se.data)

    def _ppl(self, dataDUE, epoch=-1, on_shell=False, display=False):
        start = datetime.now()
        self._log("start _ppl")

        ppl_w_log = 0
        ppl_e_log = 0
        ppl_log = 0

        Nd_sum = 0
        Md_sum = 0
        D_sum = 0

        for docdata in dataDUE.generate():
            try:
                if on_shell:
                    doc_ppl_log, Nd, Md = self._ppl_log_single_document_on_shell(docdata)
                else:
                    ### test ###
                    doc_ppl_log, Nd, Md = self._ppl_log_single_document_off_shell(docdata, display=display)
            except FloatingPointError as e:
                self._log("encounting underflow problem, no need to continue")
                return np.nan, np.nan, np.nan
            ppl_w_log += doc_ppl_log[0]
            ppl_e_log += doc_ppl_log[1]
            ppl_log += doc_ppl_log[2]
            Nd_sum += Nd
            Md_sum += Md
            D_sum += 1
        # normalize #
        ppl_w_log /= Nd_sum
        ppl_e_log /= Md_sum
        ppl_log /= D_sum

        duration = (datetime.now() - start).total_seconds()
        self._log("_ppl takes %fs" % duration)

        return ppl_w_log, ppl_e_log, ppl_log                                # word & emoti not separable

    def _ppl_log_single_document_off_shell(self, docdata, display=False):
        d, docToken, [doc_u, doc_e] = docdata
        return [0, 0, 0], docToken.shape[0], doc_u.shape[0]

    def _ppl_log_single_document_on_shell(self, docdata):
        d, docToken, [doc_u, doc_e] = docdata
        return [0, 0, 0], docToken.shape[0], doc_u.shape[0]

    def _predict_single_document_on_shell(self, docdata):
        d, docToken, [doc_u, doc_e] = docdata
        return self._predict_single_document_core(
            doc_z2=self.z2[d],
            doc_z1=self.z1[d],
            docdata=docdata
        )

    def _predict_single_document_off_shell(self, docdata):
        doc_z2, doc_z1, _ = self._partial_inference_y_z(docdata=docdata)
        return self._predict_single_document_core(
            doc_z2=doc_z2,
            doc_z1=doc_z1,
            docdata=docdata
        )

    def _partial_inference_y_z(self, docdata, max_iter_inner=50):
        d, docToken, [doc_u, doc_e] = docdata
        doc_z2 = probNormalize(self.GLV["theta2"])
        doc_z1 = probNormalize(self.GLV["theta1"])
        doc_z2_old = doc_z2.copy()
        doc_z1_old = doc_z1.copy()
        doc_y_old = np.zeros([self.Nd[d], 2])
        converge_flag = False

        for inner_iter in range(max_iter_inner):
            doc_y = self._partial_inference_y_update(
                doc_z2=doc_z2, doc_z1=doc_z1, pi_avg=self.GLV["pi"],
                phi2_avg=self.GLV["phi2"], phi1_avg=self.GLV["phi1"],
                docToken=docToken
            )
            doc_z2 = self._partial_inference_z_update(
                theta_avg=self.GLV["theta2"], doc_y_i=doc_y[:, 1], docToken=docToken, phi_avg=self.GLV["phi2"]
            )
            doc_z1 = self._partial_inference_z_update(
                theta_avg=self.GLV["theta1"], doc_y_i=doc_y[:, 0], docToken=docToken, phi_avg=self.GLV["phi1"]
            )
            doc_z2_old, doc_z1_old, doc_y_old, converge_flag, diff = self._partial_inference_convergeCheck(
                doc_z2=doc_z2, doc_z1=doc_z1, doc_y=doc_y,
                doc_z2_old=doc_z2_old, doc_z1_old=doc_z1_old, doc_y_old=doc_y_old,
                doc_Nd=self.Nd[d]
            )
            if converge_flag:
                break
        if not converge_flag:
            warnings.warn("%d document not converged after %d in partial inference" % (d, max_iter_inner))
        return doc_z2, doc_z1, doc_y

    def _partial_inference_y_update(self, doc_z2, doc_z1, pi_avg, phi2_avg, phi1_avg, docToken):
        doc_y_unnorm_log = np.zeros([docToken.shape[0], 2])
        doc_y_unnorm_log[:, 0] = np.dot(doc_z1, np.log(phi1_avg[:, docToken]))
        doc_y_unnorm_log[:, 1] = np.dot(doc_z2, np.log(phi2_avg[:, docToken]))
        doc_y_unnorm_log += np.log(pi_avg)
        return probNormalizeLog(doc_y_unnorm_log)

    def _partial_inference_z_update(self, theta_avg, doc_y_i, docToken, phi_avg):
        doc_z_unnorm_log = np.dot(
            np.log(phi_avg[:, docToken]),
            doc_y_i
        )
        doc_z_unnorm_log += np.log(theta_avg)
        return probNormalizeLog(doc_z_unnorm_log)

    def _partial_inference_convergeCheck(self, doc_z2, doc_z1, doc_y, doc_z2_old, doc_z1_old, doc_y_old, doc_Nd):
        diff_z2 = np.linalg.norm(doc_z2 - doc_z2_old) / np.sqrt(doc_z2.shape[0])
        diff_z1 = np.linalg.norm(doc_z1 - doc_z1_old) / np.sqrt(doc_z1.shape[0])
        if doc_Nd == 0:
            diff_y = 0
        else:
            diff_y = np.linalg.norm(doc_y - doc_y_old) / np.sqrt(doc_Nd * 2)
        diff_total = diff_z2 + diff_z1 + diff_y
        if diff_total < self.converge_threshold_inner:
            converge = True
        else:
            converge = False
        return doc_z2, doc_z1, doc_y, converge, diff_total

    def _predict_single_document_core(self, doc_z2, doc_z1, docdata):
        d, docToken, [doc_u, doc_e] = docdata
        prob_e_r0 = self.GLV["eta_u"][doc_u, :]
        prob_e_r1 = np.dot(doc_z1, self.GLV["eta_z"])
        prob_e_r2 = np.tensordot(
            doc_z2,
            np.tensordot(
                self.GLV["psi"][doc_u, :],
                self.GLV["eta"],
                axes=(1, 1)
            ),
            axes=(0, 1)
        )
        prob_e = self.GLV["se"][0] * prob_e_r0 + self.GLV["se"][1] * prob_e_r1 + self.GLV["se"][2] * prob_e_r2
        prob_e = probNormalize(prob_e)
        return prob_e.tolist(), doc_e.tolist()

    def _ppl_compare(self, ppl_best, ppl):
        N_ppl = len(ppl)
        new_best = [ppl_best[i] for i in range(N_ppl)]
        best_flag = [False for i in range(N_ppl)]
        for i in range(N_ppl):
            if ppl_best[i] is None:
                # no such valid data to calculate #
                continue
            if ppl_best[i] < ppl[i]:
                new_best[i] = ppl[i]
                best_flag[i] = True
        return new_best, best_flag

    def _fit_single_epoch(self, dataDUE, dataW, dataToken, epoch, batch_size):
        """ single process"""
        self._log("start _fit_single_epoch")
        start = datetime.now()

        # uniformly sampling all documents once #
        #pbar = tqdm(dataDUE.batchGenerate(batch_size=batch_size, keep_complete=True),
        #            total = math.ceil(self.D_train * 1.0 / batch_size),
        #            desc = '({0:^3})'.format(epoch))
        #for i_batch, batch_size_real, data_batched in pbar:
        for i_batch, batch_size_real, data_batched in dataDUE.batchGenerate(batch_size=batch_size, keep_complete=True):
            var_temp = self._fit_batchIntermediateInitialize()

            pars_topass = self._fit_single_epoch_pars_topass()

            ### test ###
            # returned_cursor = self.pool.imap_unordered(_fit_single_document, data_batched_topass)
            # for returned in returned_cursor:
            for data_batched_sample in data_batched:
                returned = _fit_single_document(data_batched_sample, pars_topass)
                var_temp = self._fit_single_batch_cumulate(returned, var_temp)

            # end3 = datetime.now()###
            self._fit_single_batch_global_update(var_temp, batch_size_real, epoch)
            # end4 = datetime.now()###
            # print "_fit_single_batch_global_update takes %fs" % (end4 - end3).total_seconds()###
        duration = (datetime.now() - start).total_seconds()
        self._log("_fit_single_epoch takes %fs" % duration)

    def _fit_batchIntermediateInitialize(self):
        # instantiate vars every loop #
        vars = {
            "T2I": np.zeros(self.K2, dtype=np.float64),
            "T1I": np.zeros(self.K1, dtype=np.float64),
            "YI": np.zeros(2, dtype=np.float64),
            "Y0TV": np.zeros([self.K1, self.V], dtype=np.float64),
            "Y1TV": np.zeros([self.K2, self.V], dtype=np.float64),
            "R2T2XE": np.zeros([self.K2, self.G, self.E], dtype=np.float64),
            "R1T1E": np.zeros([self.K1, self.E], dtype=np.float64),
            "R0UE": np.zeros([self.U, self.E], dtype=np.float64),
            "UX": np.zeros([self.U, self.G], dtype=np.float64),
            "RI": np.zeros(3, dtype=np.float64)
        }
        return vars

    def _fit_single_epoch_pars_topass(self):
        ans = vars(self)
        pars_topass = {}
        for name in ans:
            if name == "pool":
                continue
            if name == "process":
                continue
            pars_topass[name] = ans[name]
        return pars_topass

    def _fit_single_batch_cumulate(self, returned_fit_single_document, var_temp):
        d, doc_z2, doc_z1, doc_YI, doc_Y0TV, doc_Y1TV, doc_R2T2XE, doc_R1T1E, doc_x, doc_u, doc_e_onehot, doc_r = \
            returned_fit_single_document
        self.z2[d, :] = doc_z2[:]
        self.z1[d, :] = doc_z1[:]

        var_temp["T2I"] += doc_z2
        var_temp["T1I"] += doc_z1
        var_temp["YI"] += doc_YI
        var_temp["Y0TV"] += doc_Y0TV
        var_temp["Y1TV"] += doc_Y1TV
        var_temp["R2T2XE"] += doc_R2T2XE
        var_temp["R1T1E"] += doc_R1T1E
        var_temp["R0UE"][doc_u, :] += (doc_e_onehot * np.expand_dims(doc_r[:, 0], axis=-1))
        var_temp["UX"][doc_u, :] += doc_x
        var_temp["RI"] += np.sum(doc_r, axis=0)
        return var_temp

    def _fit_single_batch_global_update(self, var_temp, batch_size_real, epoch):
        lr = self._lrCal(epoch)
        batch_weight = self.D_train * 1.0 / batch_size_real
        new_theta2_temp = self.alpha2 + batch_weight * var_temp["T2I"]
        new_theta1_temp = self.alpha1 + batch_weight * var_temp["T1I"]
        new_pi_temp = self.delta + batch_weight * var_temp["YI"]
        new_phi1_temp = self.beta + batch_weight * var_temp["Y0TV"]
        new_phi2_temp = self.beta + batch_weight * var_temp["Y1TV"]
        new_eta_temp = self.gamma + batch_weight * var_temp["R2T2XE"]
        new_eta_z_temp = self.gamma_z + batch_weight * var_temp["R1T1E"]
        new_eta_u_temp = self.gamma_u + batch_weight * var_temp["R0UE"]
        new_psi_temp = self.zeta + batch_weight * var_temp["UX"]
        new_se_temp = self.omega + batch_weight * var_temp["RI"]

        self.theta2.update(new_theta2_temp, lr)
        self.theta1.update(new_theta1_temp, lr)
        self.pi.update(new_pi_temp, lr)
        self.phi1.update(new_phi1_temp, lr)
        self.phi2.update(new_phi2_temp, lr)
        self.eta.update(new_eta_temp, lr)
        self.eta_z.update(new_eta_z_temp, lr)
        self.eta_u.update(new_eta_u_temp, lr)
        self.psi.update(new_psi_temp, lr)
        self.se.update(new_se_temp, lr)

    def _lrCal(self, epoch):
        return float(self.lr["init"] * np.power((self.lr["tau"] + epoch), - self.lr["kappa"]))

    def _log(self, string):
        with open(self.log_file, "a") as logf:
            logf.write(string.rstrip("\n") + "\n")

    def _restoreCheckPoint(self, filename=None):
        start = datetime.now()

        if filename is None:
            filename = self.checkpoint_file
        state = cPickle.load(open(filename, "rb"))
        # restore #
        self.theta2.initialize(new_data=state["theta2"])
        self.theta1.initialize(new_data=state["theta1"])
        self.pi.initialize(new_data=state["pi"])
        self.eta.initialize(new_data=state["eta"])
        self.eta_z.initialize(new_data=state["eta_z"])
        self.eta_u.initialize(new_data=state["eta_u"])
        self.phi1.initialize(new_data=state["phi1"])
        self.phi2.initialize(new_data=state["phi2"])
        self.psi.initialize(new_data=state["psi"])
        self.se.initialize(new_data=state["se"])
        self.alpha2 = state["alpha2"]
        self.alpha1 = state["alpha1"]
        self.beta = state["beta"]
        self.gamma = state["gamma"]
        self.gamma_z = state["gamma_z"]
        self.gamma_u = state["gamma_u"]
        self.delta = state["delta"]
        self.zeta = state["zeta"]
        self.omega = state["omega"]
        self.z2 = state["z2"]
        self.z1 = state["z1"]
        self.epoch_init = state["epoch"] + 1
        ppl = state["ppl"]
        print "restore state from file '%s' on epoch %d with ppl: %s" % (filename, state["epoch"], str(ppl))

        duration = datetime.now() - start
        print "_restoreCheckPoint takes %f s" % duration.total_seconds()

        # for model display #
        self._estimateGlobal()

    def _saveCheckPoint(self, epoch, ppl, perf, filename=None):
        start = datetime.now()

        if filename is None:
            filename = self.checkpoint_file
        state = {
            "theta2": self.theta2.data,
            "theta1": self.theta1.data,
            "pi": self.pi.data,
            "eta": self.eta.data,
            "eta_z": self.eta_z.data,
            "eta_u": self.eta_u.data,
            "phi1": self.phi1.data,
            "phi2": self.phi2.data,
            "psi": self.psi.data,
            "se": self.se.data,
            "alpha2": self.alpha2,
            "alpha1": self.alpha1,
            "beta": self.beta,
            "gamma": self.gamma,
            "gamma_z": self.gamma_z,
            "gamma_u": self.gamma_u,
            "delta": self.delta,
            "zeta": self.zeta,
            "omega": self.omega,
            "z2": self.z2,
            "z1": self.z1,
            "epoch": epoch,
            "ppl": ppl,
            "perf": perf
        }
        with open(filename, "wb") as f_ckpt:
            cPickle.dump(state, f_ckpt)

        duration = datetime.now() - start
        self._log("_saveCheckPoint takes %f s" % duration.total_seconds())


def _fit_single_document(docdata, pars_topass, max_iter_inner=500):
    d, docToken, [doc_u, doc_e] = docdata
    doc_Nd = pars_topass["Nd"][d]
    doc_Md = pars_topass["Md"][d]

    # random initialization #
    doc_z2 = probNormalize(np.random.random([pars_topass["K2"]]))
    doc_z1 = probNormalize(np.random.random([pars_topass["K1"]]))
    doc_r = probNormalize(np.ones([doc_Md, 3], dtype=np.float64))
    # old for comparison #
    doc_x_old = np.zeros([doc_Md, pars_topass["G"]])
    doc_y_old = np.zeros([doc_Nd, 2])
    doc_r_old = doc_r.copy()
    doc_z2_old = doc_z2.copy()
    doc_z1_old = doc_z1.copy()
    converge_flag = False

    for inner_iter in range(max_iter_inner):
        doc_y = _fit_single_document_y_update(
            doc_z2=doc_z2, doc_z1=doc_z1, docToken=docToken, pars_topass=pars_topass
        )
        doc_x = _fit_single_document_x_update(
            doc_z2=doc_z2, doc_r=doc_r, doc_u=doc_u, doc_e=doc_e, pars_topass=pars_topass
        )
        doc_r = _fit_single_document_r_update(
            doc_z2=doc_z2, doc_z1=doc_z1, doc_x=doc_x, doc_u=doc_u, doc_e=doc_e, pars_topass=pars_topass
        )
        doc_z2 = _fit_single_document_z2_update(
            doc_x=doc_x, doc_y=doc_y, doc_r=doc_r, docToken=docToken, doc_e=doc_e, pars_topass=pars_topass
        )
        doc_z1 = _fit_single_document_z1_update(
            doc_y=doc_y, doc_r=doc_r, docToken=docToken, doc_e=doc_e, pars_topass=pars_topass
        )
        doc_x_old, doc_y_old, doc_r_old, doc_z2_old, doc_z1_old, converge_flag, diff = \
            _fit_single_document_convergeCheck(
                doc_x=doc_x,
                doc_y=doc_y,
                doc_r=doc_r,
                doc_z2=doc_z2,
                doc_z1=doc_z1,
                doc_x_old=doc_x_old,
                doc_y_old=doc_y_old,
                doc_r_old=doc_r_old,
                doc_z2_old=doc_z2_old,
                doc_z1_old=doc_z1_old,
                pars_topass=pars_topass
            )
        if converge_flag:
            break
    if not converge_flag:
        warnings.warn("%d document not converged after %d" % (d, max_iter_inner))
    return _fit_single_document_return(
        d=d, doc_x=doc_x, doc_y=doc_y, doc_r=doc_r, doc_z2=doc_z2, doc_z1=doc_z1,
        docToken=docToken, doc_u=doc_u, doc_e=doc_e, pars_topass=pars_topass
    )


def _fit_single_document_y_update(doc_z2, doc_z1, docToken, pars_topass):
    doc_y_unnorm_log = np.zeros([docToken.shape[0], 2])
    doc_y_unnorm_log[:, 0] = np.dot(doc_z1, pars_topass["phi1"].bigamma_data[:, docToken])
    doc_y_unnorm_log[:, 1] = np.dot(doc_z2, pars_topass["phi2"].bigamma_data[:, docToken])
    doc_y_unnorm_log += pars_topass["pi"].bigamma_data
    return probNormalizeLog(doc_y_unnorm_log)


def _fit_single_document_x_update(doc_z2, doc_r, doc_u, doc_e, pars_topass):
    doc_x_unnorm_log_u = pars_topass["psi"].bigamma_data[doc_u]
    doc_x_unnorm_log_e = np.expand_dims(doc_r[:, 2], axis=-1) * np.transpose(
        np.tensordot(
            doc_z2,
            pars_topass["eta"].bigamma_data,
            axes=(0, 0)
        ),
        axes=(1, 0)
    )[doc_e, :]
    doc_x_unnorm_log = doc_x_unnorm_log_u + doc_x_unnorm_log_e
    return probNormalizeLog(doc_x_unnorm_log)


def _fit_single_document_r_update(doc_z2, doc_z1, doc_x, doc_u, doc_e, pars_topass):
    Md = doc_u.shape[0]
    doc_r_unnorm_log = np.zeros([Md, 3])
    doc_r_unnorm_log[:, 0] = pars_topass["eta_u"].bigamma_data[doc_u, doc_e]
    doc_r_unnorm_log[:, 1] = np.dot(doc_z1, pars_topass["eta_z"].bigamma_data[:, doc_e])
    doc_r_unnorm_log[:, 2] = np.tensordot(
        doc_x,
        np.tensordot(
            doc_z2,
            pars_topass["eta"].bigamma_data,
            axes=(0, 0)
        ),
        axes=(1, 0)
    )[np.arange(Md), doc_e]
    doc_r_unnorm_log += pars_topass["se"].bigamma_data
    return probNormalizeLog(doc_r_unnorm_log)


def _fit_single_document_z2_update(doc_x, doc_y, doc_r, docToken, doc_e, pars_topass):
    doc_z2_unnorm_log_w = np.dot(
        pars_topass["phi2"].bigamma_data[:, docToken],
        doc_y[:, 1]
    )
    doc_z2_unnorm_log_e = np.tensordot(
        pars_topass["eta"].bigamma_data[:, :, doc_e] * doc_r[:, 2],
        doc_x,
        axes=([2, 1], [0, 1])
    )
    doc_z2_unnorm_log_theta2 = pars_topass["theta2"].bigamma_data
    doc_z2_unnorm_log = doc_z2_unnorm_log_w + doc_z2_unnorm_log_e + doc_z2_unnorm_log_theta2
    return probNormalizeLog(doc_z2_unnorm_log)


def _fit_single_document_z1_update(doc_y, doc_r, docToken, doc_e, pars_topass):
    doc_z1_unnorm_log_w = np.dot(
        pars_topass["phi1"].bigamma_data[:, docToken],
        doc_y[:, 0]
    )
    doc_z1_unnorm_log_e = np.tensordot(
        doc_r[:, 1],
        pars_topass["eta_z"].bigamma_data[:, doc_e],
        axes=[0, -1]
    )
    doc_z1_unnorm_log_theta1 = pars_topass["theta1"].bigamma_data
    doc_z1_unnorm_log = doc_z1_unnorm_log_w + doc_z1_unnorm_log_e + doc_z1_unnorm_log_theta1
    return probNormalizeLog(doc_z1_unnorm_log)


def _fit_single_document_convergeCheck(
        doc_x, doc_y, doc_r, doc_z2, doc_z1, doc_x_old, doc_y_old, doc_r_old, doc_z2_old, doc_z1_old,
        pars_topass=None
):
    """ simple square difference check"""
    doc_Md = doc_x.shape[0]
    doc_Nd = doc_y.shape[0]
    diff_x = np.linalg.norm(doc_x - doc_x_old) / np.sqrt(doc_Md * pars_topass["G"])
    diff_r = np.linalg.norm(doc_r - doc_r_old) / np.sqrt(doc_Md * 3)
    if doc_Nd == 0:
        diff_y = 0
    else:
        diff_y = np.linalg.norm(doc_y - doc_y_old) / np.sqrt(doc_Nd * 2)
    diff_z2 = np.linalg.norm(doc_z2 - doc_z2_old) / np.sqrt(pars_topass["K2"])
    diff_z1 = np.linalg.norm(doc_z1 - doc_z1_old) / np.sqrt(pars_topass["K1"])
    diff_total = diff_x + diff_y + diff_r + diff_z2 + diff_z1
    if diff_total < pars_topass["converge_threshold_inner"]:
        converge = True
    else:
        converge = False
    return doc_x, doc_y, doc_r, doc_z2, doc_z1, converge, diff_total


def _fit_single_document_return(d, doc_x, doc_y, doc_r, doc_z2, doc_z1, docToken, doc_u, doc_e, pars_topass):
    Nd = docToken.shape[0]
    Md = doc_e.shape[0]
    docToken_onehot = np.zeros([Nd, pars_topass["V"]])
    docToken_onehot[np.arange(Nd), docToken] = 1
    doc_e_onehot = np.zeros([Md, pars_topass["E"]])
    doc_e_onehot[np.arange(Md), doc_e] = 1

    doc_YI = np.sum(doc_y, axis=0)
    doc_Y0TV = np.outer(doc_z1, np.dot(doc_y[:, 0], docToken_onehot))
    doc_Y1TV = np.outer(doc_z2, np.dot(doc_y[:, 1], docToken_onehot))
    doc_R2T2XE = np.einsum(
        "i,jk->ijk",
        doc_z2,
        np.tensordot(
            doc_x,
            doc_e_onehot * np.expand_dims(doc_r[:, 2], axis=-1),
            axes=(0, 0)
        )
    )
    doc_R1T1E = np.einsum(
        "i,j->ij",
        doc_z1,
        np.dot(
            doc_r[:, 1],
            doc_e_onehot
        )
    )
    return d, doc_z2, doc_z1, doc_YI, doc_Y0TV, doc_Y1TV, doc_R2T2XE, doc_R1T1E, doc_x, doc_u, doc_e_onehot, doc_r
