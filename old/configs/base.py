
from easydict import EasyDict

def base_config():
    conf = EasyDict
    conf.backbone = 'resnet50'
    conf.embedding_size = 128
    conf.metric = 'arcface'
    conf.num_classes = 12345
    conf.criterion = 'softmax'
    
    conf.optimizer = 'sgd'

    conf.learning_rate = 0.01
    conf.weight_decay = 0.01
    conf.learning_rate_step = 0.01

    conf.number_of_epochs = 50

    conf.dataset = 'iris'

    return conf