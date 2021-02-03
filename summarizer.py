
import numpy as np
import os
import pandas as pd
import config_sum as cf
import matplotlib.pyplot as plt


class Summary:
    def __init__(self, cnf):
        self.cnf = cnf
        self.path_out = self.cnf.path_out + '/fig'
        if not os.path.isdir(self.path_out):
            os.makedirs(self.path_out)
        
        #plt.rcParams["font.size"] = 14


    def outGraph(self):
        fig = plt.figure(figsize=(12, 5))
        ax  = fig.add_subplot(1,1,1)
        for i in range(len(self.cnf.log_name)):
            path_dat = self.cnf.path_out + '/_result_' + self.cnf.log_name[i] + '/statistics.csv'
            if os.path.exists(path_dat):
                dat = pd.read_csv(path_dat, index_col = 0)
                if self.cnf.mode == "med":
                    ax.plot(dat.index, dat['med'] , linestyle='solid', color = self.cnf.color[i], label= self.cnf.log_name[i]) 
                    ax.fill_between(dat.index, dat['q25'], dat['q75'], facecolor=self.cnf.color[i], alpha=0.1)
                elif self.cnf.mode == "ave":
                    ax.plot(dat.index, dat['ave'] , linestyle='solid', color = self.cnf.color[i], label= self.cnf.log_name[i]) 
        #ax.set_xlim({0, self.cnf.max_epoch})
        ax.set_xlim({0, self.cnf.max_epoch})
        ax.set_ylabel('RMSE')
        ax.set_xlabel('Epoch')
        #ax.set_title()

        ax.legend(self.cnf.log_name)        

        plt.subplots_adjust(left=0.075, right=0.96, bottom=0.11, top=0.94)

        #plt.yscale('log')

        fig.savefig(self.path_out + "/compare_AE_i10_e10000_" + self.cnf.mode + ".png" , dpi=150)

    def getTestScore(self):
        _out = []
        for i in range(len(self.cnf.log_name)):
            tmp = []
            for j in range(self.cnf.max_trial):
                path_dat = self.cnf.path_out + '/_result_' + self.cnf.log_name[i] + '/trials/trial' + str(j) + '_test.csv'
                if os.path.exists(path_dat):
                    dat = pd.read_csv(path_dat, index_col = 0)
                    tmp.append(dat.iat[0,0])
            _out.append(np.mean(tmp))    
        print("score:", _out)

if __name__ == '__main__':
    cnf = cf.Configuration()
    smy = Summary(cnf)
    # smy.outGraph()
    smy.getTestScore()
    print("done")
