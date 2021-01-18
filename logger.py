
import os
import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt


class Logger:
    def __init__(self, cnf):
        self.dat, self.cnf = [], cnf
        self.path_out = cnf.path_out
        self.path_out += '/{0}/'.format(self.cnf.log_name)
        self.path_trial = self.path_out + 'trials' 
        if not os.path.isdir(self.path_trial):
            os.makedirs(self.path_trial)

    def logging(self, epo, error, hid_w, out_w):
        sls = [epo, error]
        sls.extend(hid_w.flatten())
        sls.extend(out_w.flatten())
        self.dat.append(sls)

    def logging_detail(self, epo, Y, error):
        dat_dtl = []
        for i in range(self.cnf.N):
            sls_dtl = [i+1, error]
            sls_dtl.extend(self.cnf.X[i])
            sls_dtl.extend(Y[i])
            dat_dtl.append(sls_dtl)

        head = "N,error," + ','.join(["X{}".format(i) for i in range(self.cnf.X.shape[1])]) + ',' + ','.join(["Y{}".format(i) for i in range(Y.shape[1])])
        np.savetxt(self.path_trial +'/trial{}_epo{}.csv'.format(self.cnf.seed, epo), np.array(dat_dtl), delimiter=',', header = head)
        print("trial: {:03}\tepoch: {}\terror: {}".format(self.cnf.seed, epo, error))


    def outLog(self):
        head = "epoch,error," + ','.join(["hid_w{}".format(i) for i in range(self.cnf.hid_lay*(self.cnf.inp_lay+1))]) + ',' + ','.join(["out_w{}".format(i) for i in range(self.cnf.out_lay*(self.cnf.hid_lay+1))])
        np.savetxt(self.path_trial +'/trial{}.csv'.format(self.cnf.seed), np.array(self.dat), delimiter=',', header = head)      
        self.dat = []


    def logging_predict(self, Y, error):
        dat_pre = []
        for i in range(self.cnf.N):
            sls_pre = [i+1, error]
            sls_pre.extend(self.cnf.test_data[i])
            sls_pre.extend(Y[i])
            dat_pre.append(sls_pre)

        head = "N,error," + ','.join(["test{}".format(i) for i in range(self.cnf.test_data.shape[1])]) + ',' + ','.join(["Y{}".format(i) for i in range(Y.shape[1])])
        np.savetxt(self.path_trial +'/trial{}_test.csv'.format(self.cnf.seed), np.array(dat_pre), delimiter=',', header = head)
        print("trial: {:03}\ttest \t\terror: {}".format(self.cnf.seed, error))



class Statistics:
    def __init__(self, cnf, path_out, path_dat):
        self.path_out = path_out
        self.path_dat = path_dat
        self.cnf      = cnf

    def outStatistics(self):
        df = None
        for i in range(self.cnf.max_trial):    
            dat = pd.read_csv(self.path_dat+'/trial{}.csv'.format(i+1), index_col = 0)
            if i == 0:
                df = pd.DataFrame({'trial{}'.format(i+1) : np.array(dat['error'])}, index = dat.index)
            else:
                df['trial{}'.format(i+1)] = np.array(dat['error'])
        df.to_csv(self.path_out + "all_trials.csv")

        _min, _max, _q25, _med, _q75, _ave, _std = [], [], [], [], [], [], []
        for i in range(len(df.index)):
            dat = np.array(df.loc[df.index[i]])
            res = np.percentile(dat, [25, 50, 75])
            _min.append(dat.min())
            _max.append(dat.max())
            _q25.append(res[0])
            _med.append(res[1])
            _q75.append(res[2])
            _ave.append(dat.mean())
            _std.append(dat.std())

        _out = pd.DataFrame({
            'min' : np.array(_min),
            'q25' : np.array(_q25),
            'med' : np.array(_med),
            'q75' : np.array(_q75),
            'max' : np.array(_max),
            'ave' : np.array(_ave),
            'std' : np.array(_std)
            },index = df.index)
        _out.to_csv(self.path_out + "statistics.csv")

