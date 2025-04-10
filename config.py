import configparser


class Config:
    def __init__(self, path):
        self.path = path
        self._load_config()

    def _load_config(self):
        parser = configparser.ConfigParser()
        parser.read(self.path)

        cfg = parser['Configure']

        # Load configuration values from .ini file
        self.num_users = int(cfg['num_users'])
        self.num_items = int(cfg['num_items'])
        self.dimension = int(cfg['dimension'])
        self.learning_rate = float(cfg['learning_rate'])
        self.epochs = int(cfg['epochs'])
        self.num_negatives = int(cfg['num_negatives'])
        self.num_evaluate = int(cfg['num_evaluate'])
        self.num_procs = int(cfg['num_procs'])
        self.topk = int(cfg['topk'])
        self.evaluate_batch_size = int(cfg['evaluate_batch_size'])
        self.training_batch_size = int(cfg['training_batch_size'])
        self.epoch_notice = int(cfg['epoch_notice'])
        self.pretrain_flag = int(cfg['pretrain_flag'])
        self.pre_model = cfg['pre_model']
        self.data_name = cfg['data_name']
        self.model_name = cfg['model_name']
        self.user_review_vector_matrix = cfg['user_review_vector_matrix']
        self.item_review_vector_matrix = cfg['item_review_vector_matrix']
        self.input_review_dim = int(cfg['input_review_dim'])
