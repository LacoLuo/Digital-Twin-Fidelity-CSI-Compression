class configurations(object):
    def __init__(self):
        # Dataset params.
        self.data_root = "DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_1_material"
        self.test_data_root = "DeepMIMO\DeepMIMO_datasets\Boston5G_3p5_real"
        self.train_csv = "train_data_idx.csv"
        self.test_csv = "test_data_idx.csv"

        # Train params.
        self.num_train_data = 5120
        self.batch_size = 32
        self.enc_dim = 32
        self.learning_rate = 1e-2
        self.num_epochs = 160
        self.gpu = 0