"""Explicit paper optimizer selection, shared by all maintained clients."""

def add_paper_arguments(parser):
    parser.set_defaults(lr=None)
    parser.add_argument('--optimizer',choices=['auto','sgd','adam'],default='auto')
    parser.add_argument('--momentum',type=float,default=.9)
    parser.add_argument('--data_root',default='data',help='Optional canonical NPZ dataset root')
    parser.add_argument('--human_labels',default=None,help='CIFAR-N safe NPZ annotation file in Keras training order')
    parser.add_argument('--human_noise_key',default=None,help='Explicit human annotation field; no inferred noise key')
    return parser


def resolve_optimizer(args):
    if args.optimizer=='auto':
        args.optimizer='sgd' if args.dataset_name in ('cifar10','cifar10n','cifar100','cifar100n','fashionmnist') else 'adam'
    if args.lr is None:args.lr=.1 if args.optimizer=='sgd' else .001
    if args.lr<=0 or not 0<=args.momentum<1:raise ValueError('lr must be positive and momentum in [0,1)')
    return args


def make_optimizer(config):
    import tensorflow as tf
    if config.get('optimizer','adam')=='sgd':
        return tf.keras.optimizers.SGD(learning_rate=config['lr'],momentum=config.get('momentum',.9))
    return tf.keras.optimizers.Adam(learning_rate=config['lr'])
