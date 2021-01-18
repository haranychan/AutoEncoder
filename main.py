import numpy            as np
import configuration    as cf
import auto_encoder     as ae
import logger           as lg
from joblib             import Parallel, delayed

def run(opt, cnf, i):
    if cnf.parallel:
        cnf.setRandomSeed(seed=i+1)

    opt.initialization()
    opt.training()
    # opt.out_errorgraph()
    opt.predict()


if __name__ == '__main__':
    cnf = cf.Configuration()
    cnf.outSetting()
    log = lg.Logger(cnf)
    if cnf.parallel:
        opt = ae.AE(cnf, log) 
        Parallel(n_jobs=-1)([delayed(run)(opt, cnf, i) for i in range(cnf.max_trial)])
    else:
        for i in range(cnf.max_trial):
            cnf.setRandomSeed(seed=i+1)
            opt = ae.AE(cnf, log) 
            run(opt, cnf, 0)
            del opt
    sts = lg.Statistics(cnf, log.path_out, log.path_trial)
    sts.outStatistics()