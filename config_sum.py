import numpy as np

class Configuration:

    def __init__(self):

        self.path_out   = "./"

        self.max_epoch  = 10000
        self.max_trial  = 5

        self.mode = "med" #[med, ave]

        self.log_name   = [ 
            "AE_i100-h10",
            "AE_i100-h25",
            "AE_i100-h50"
        ]

        # self.log_name   = [ 
        #     "AE_i50-h5",
        #     "AE_i50-h10",
        #     "AE_i50-h25",
        #     "AE_i50-h40"
        # ]

        # self.log_name   = [ 
        #     "AE_i25-h5",
        #     "AE_i25-h10",
        #     "AE_i25-h15",
        #     "AE_i25-h20"
        # ]

        # self.log_name   = [ 
        #     "AE_i10-h5",
        #     "AE_i10-h10"
        # ]

        self.color      = [
            "blue",
            "red",
            "green",
            "orange",
            "purple"
        ]
