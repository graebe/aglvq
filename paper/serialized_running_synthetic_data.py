# Imports
from aglvq.dataset import create_dataset
from aglvq.model import Alvq_Model
from keras.utils import to_categorical

data_list = [
    # 'const',
    'variable',
    # 'nonLinear'
    ]
importances = [
    'none',
    # 'relevance',
    # 'matrix',
    # 'tangent'
    ]
proto_types = [
    # 'const',
    'polynom',
    # 'neural'
    ]
config = {
    # Lr, lr_pre, reg_rate, epochs, epochs_pre
    'const': {
        'none': [5e-5, 1e-3, 0, 40, 40],
        'relevance': [1e-5, 1e-3, 0, 60, 40],
        'matrix': [1e-5, 1e-3, 0, 80, 40],
        'tangent': [1e-5, 1e-3, 0, 80, 40]},
    'polynom': {
        'none': [3e-3, 3e-3, 0, 30, 70],
        'relevance': [1e-3, 3e-3, 1e-9, 100, 100],
        'matrix': [1e-3, 3e-3, 1e-9, 100, 100],
        'tangent': [1e-3, 3e-3, 1e-9, 100, 100]},
    'neural': {
        'none': [1e-4, 6e-4, 1e-7, 90, 30],
        'relevance': [9e-6, 1e-4, 1e-9, 200, 60],
        'matrix': [1e-5, 6e-4, 1e-9, 200, 30],
        'tangent': [6e-6, 2e-4, 1e-9, 250, 100]}}


scores = dict()
for dataset in data_list:
    # Data
    x_train, y_train, proto_input_train, x_test, y_test, proto_input_test = \
        create_dataset(set_type=dataset)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    for p_type in proto_types:
        for imp in importances:
            title = None
            title = dataset + '_' + p_type + '_' + imp

            # Setup Model
            classifier = Alvq_Model(ptype=p_type,
                                    imptype=imp,
                                    prototype_shape=x_train.shape[1:],
                                    env_shape=(1,),
                                    n_classes=3,
                                    n_poly=2,
                                    learning_rate=config[p_type][imp][0],
                                    reg_rate=config[p_type][imp][2],
                                    pre_train_lr=config[p_type][imp][1],
                                    initialize_data=[x_train, y_train],
                                    pre_training=True,
                                    n_tangents=10)
            if p_type == 'const':
                classifier.fit(x_train, y_train,
                               epochs=config[p_type][imp][3],
                               epochs_pre_train=config[p_type][imp][4],
                               batch_size=32)
                scores[title] = classifier.score(x_test, y_test)[-1]
                print(title)
            else:
                classifier.fit([x_train, proto_input_train], y_train,
                               epochs=config[p_type][imp][3],
                               epochs_pre_train=config[p_type][imp][4],
                               batch_size=32,
                               )
                scores[title] = classifier.score([x_test, proto_input_test],
                                                 y_test)[-1]
                print(title)

            if p_type == 'const':
                classifier.plot_prototypes_2d(
                    fname=title,
                    xlabel='Features [-]',
                    ylabel='Values [-]',
                    title='',
                    save=True)
            else:
                classifier.plot_prototypes_2d(
                    env=proto_input_test,
                    xlabel='Features [-]',
                    ylabel='Values [-]',
                    fname=title,
                    title='',
                    save=True)

            classifier.plot_training_curves()
            classifier.plot_training_curves(pre_train=True)
            classifier = None
