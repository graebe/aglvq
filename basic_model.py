# Imports
from aglvq.dataset import create_dataset
from aglvq.model import Alvq_Model
from aglvq import model
from keras.utils import to_categorical

# Data
X_train, Y_train, C_train, X_test, Y_test, C_test = \
    create_dataset(set_type='variable')
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Setup Model
classifier = Alvq_Model(ptype='polynom',
                        imptype='matrix',
                        prototype_shape=X_train.shape[1:],
                        env_shape=(1,),
                        n_classes=3,
                        n_poly=3,
                        pre_train_lr=0.001,
                        learning_rate=0.0001,
                        initialize_data=[X_train, Y_train],
                        pre_training=True,
                        aux_factor=1)

# Plot model
# classifier.plot_model('basic_model.png')

# Training
classifier.fit([X_train, C_train], Y_train, epochs=1000, epochs_pre_train=2000)

# Prediction
prediction = classifier.predict([X_test, C_test])
distances = classifier.get_distances([X_train, C_train])
score = classifier.score([X_test, C_test], Y_test)
# Plot
classifier.plot_prototypes_3d('test', env=C_test, multi=True)
classifier.plot_distance_histogram([X_test, C_test], Y_test)
classifier.plot_training_curves(pre_train=True)
classifier.plot_training_curves()
classifier.plot_importances()
classifier.plot_feature_values(C_test, [X_test, Y_test], n_features=[1, 3], save=True)
