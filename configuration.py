import numpy as np
import os.path
import datetime, shutil

class Configuration:

    def __init__(self):

        # Experimental setting 
        self.max_trial  = 5
        self.max_epoch  = 10000
        self.parallel   = True
        self.rd         = np.random
        self.rd.seed(0)

        # NN setting
        self.inp_lay    = int(input("The number of input layer nodes (data length): "))
        self.hid_lay    = int(input("The number of hidden layer nodes: "))
        self.out_lay    = self.inp_lay

        self.epsilon    = 0.1   # learning rate [0,1]
        self.mu         = 0.9   # momentum coefficient [0,1]

        # Data setting 
        self.N          = 100   # the number of data
        self.X          = np.array([[self.rd.randint(2) for i in range(self.inp_lay)] for n in range(self.N)])
        self.test_data  = np.array([[self.rd.randint(2) for i in range(self.inp_lay)] for n in range(self.N)])

        # I/O setting
        self.path_out   = "./"
        now = datetime.datetime.now()
        self.log_name   = "_result_" + "AE" +\
            "_" + str(now.year) +\
                "-" + str(now.month) +\
                    "-" + str(now.day) +\
                        "-" + str(now.hour) +\
                            "-" + str(now.minute)



    def setRandomSeed(self, seed=1):
        self.seed = seed
        self.rd = np.random
        self.rd.seed(self.seed)



    # out config in txt
    def outSetting(self):
    
        body_setting = "+++++ Experimental Setting +++++\n"
        body_setting += "\n< Environmental Setting >\n"

        # Environment Setting
        item_env = ["trials", "epoch", "input layer node", "hidden layer node", "output layer node", "epsilon", "mu"]
        val_env = [self.max_trial, self.max_epoch, self.inp_lay, self.hid_lay, self.out_lay, self.epsilon, self.mu]

        for i in range(len(item_env)):
            body_setting += item_env[i].ljust(12) + ": " + str(val_env[i]) + "\n"

        path_out = self.path_out + self.log_name + "/"

        # save
        if not os.path.isdir(path_out):
            os.makedirs(path_out)
        with open( path_out +"experimental_setting.txt", "w") as f:
            f.write(body_setting)
