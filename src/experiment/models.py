import tensorflow as tf
import keras.backend as K
# from keras.engine.topology import Layer
from keras.metrics import binary_accuracy
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout, Layer
from keras.models import Model
from keras import regularizers


def binary_classification_loss(concat_true, concat_pred): # used during training to optimize model
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002 # avoid log(0)
    losst = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred)) # binary cross-entropy because treatment is binary

    return losst


def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_pred)) # sqrd error for control group
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred)) #sqrd error for treatment group

    return loss0 + loss1


def ned_loss(concat_true, concat_pred): # what's the difference between this and binary_classification_loss?
    t_true = concat_true[:, 1]

    t_pred = concat_pred[:, 1]
    return tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))


def dead_loss(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred)


def dragonnet_loss_binarycross(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)


def treatment_accuracy(concat_true, concat_pred): # measures model performance; % of correct treatment predictions
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    return binary_accuracy(t_true, t_pred)



def track_epsilon(concat_true, concat_pred): # absolute mean of epsilon values --> used in targeted regularization
    epsilons = concat_pred[:, 3]
    return tf.abs(tf.reduce_mean(epsilons))


class EpsilonLayer(Layer): # custom layer to learn epsilon parameter for targeted regularization

    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 1],
                                       initializer='RandomNormal',
                                       #  initializer='ones',
                                       trainable=True)
        super(EpsilonLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        # import ipdb; ipdb.set_trace()
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]

""" 
Reason for nested function below:
- Keras requires loss functions to have exactly two arguments: y_true and y_pred
- However, we want to create a loss function that can be customized with a ratio parameter
- Therefore, we define a function make_tarreg_loss that takes the ratio parameter and returns
  a new loss function tarreg_ATE_unbounded_domain_loss that has the required signature.
For future reference of an alternative approach: https://stackoverflow.com/questions/46858016/keras-custom-loss-function-to-pass-arguments-other-than-y-true-and-y-pred  

"""
def make_tarreg_loss(ratio=1., dragonnet_loss=dragonnet_loss_binarycross):
    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]

        epsilons = concat_pred[:, 3]
        t_pred = (t_pred + 0.01) / 1.02
        # t_pred = tf.clip_by_value(t_pred,0.01, 0.99,name='t_pred')

        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred) # formula in the paper

        y_pert = y_pred + epsilons * h # perturbed prediction
        targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))

        # final
        loss = vanilla_loss + ratio * targeted_regularization
        return loss

    return tarreg_ATE_unbounded_domain_loss


def make_dragonnet(input_dim, reg_l2):
    """
    Neural net predictive model. The dragon has three heads.
    :param input_dim:
    :param reg:
    :return:
    """
    t_l1 = 0.
    t_l2 = reg_l2
    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(inputs) # elu = exponential linear unit
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x) # second layer
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)


    t_predictions = Dense(units=1, activation='sigmoid')(x)

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')
    # logging.info(epsilons)
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
    model = Model(inputs=inputs, outputs=concat_pred)

    return model


def make_tarnet(input_dim, reg_l2):
    """
    Neural net predictive model. The dragon has three heads.
    :param input_dim:
    :param reg:
    :return:
    """

    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)

    t_predictions = Dense(units=1, activation='sigmoid')(inputs)

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')
    # logging.info(epsilons)
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
    model = Model(inputs=inputs, outputs=concat_pred)

    return model


def make_ned(input_dim, reg_l2=0.01):
    """
    Neural net predictive model. The dragon has three heads.
    :param input_dim:
    :param reg:
    :return:
    """

    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='ned_hidden1')(inputs)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='ned_hidden2')(x)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='ned_hidden3')(x)

    t_predictions = Dense(units=1, activation='sigmoid', name='ned_t_activation')(x)
    y_predictions = Dense(units=1, activation=None, name='ned_y_prediction')(x)

    concat_pred = Concatenate(1)([y_predictions, t_predictions])

    model = Model(inputs=inputs, outputs=concat_pred)
    return model


def post_cut(nednet, input_dim, reg_l2=0.01): # builds model on top of frozen nednet
    for layer in nednet.layers:
        layer.trainable = False
    nednet.layers.pop()
    nednet.layers.pop()
    nednet.layers.pop() # remove last 3 layers (predictions)

    frozen = nednet

    x = frozen.layers[-1].output # get output of the representation layer
    frozen.layers[-1].outbound_nodes = []
    input = frozen.input # the frozen NED network's input layer that we will reuse

    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='post_cut_y0_1')(x)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='post_cut_y1_1')(x)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='post_cut_y0_2')(
        y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='post_cut_y1_2')(
        y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    concat_pred = Concatenate(1)([y0_predictions, y1_predictions])

    model = Model(inputs=input, outputs=concat_pred)
    return model

