
from keras import layers, models, optimizers
from keras import backend as K
import numpy as np

class Critic:
  #critic (value) model

  def __init__(self, state_size, action_size):
    #initialise parameters and build model

    self.state_size = state_size
    self.action_size = action_size

    #initialise any other variables
    self.learning_rate = 0.001

    self.build_model()

  def build_model(self):
    #build a critic (value) network that maps (state, action) pairs to Q-values

    #define input layers
    states = layers.Input(shape=(self.state_size,), name='states')
    actions = layers.Input(shape=(self.action_size,), name='actions')

    #add hidden layer(s) for state pathway

    net_states = layers.Dense(units=400, kernel_regularizer=layers.regularizers.l2(1e-6))(states)
    net_states = layers.Batchnormalization()(net_states)
    net_states = layers.Activation('relu')(net_states)

    #add hidden layer(s) for action pathway
    net_states = layers.Dense(units=300, kernel_regularizer=layers.regularizers.l2(1e-6))(net_states)
    net_actions = layers.Dense(units=300, kernel_regularizer=layers.regularizers.l2(1e-6))(actions)

    #combine state and action pathways

    net = layers.Add()([net_states, net_actions])
    net = layers.Activation('relu')(net)

    #add final output layer to produce action values (Q values)
    Q_values = layers.Dense(units=1, name='q_values', kernel_initializer = layers.initializers.RandomUniform(minval=-0.003, maxval=0.003))(net)

    #create keras model
    self.model = models.Model(inputs=[states, actions], outputs=Q_values)

    #define optimizer and compile model for training with built-in loss function
    optimizer = optimizers.Adam(lr=self.learning_rate)
    self.model.compile(optimizer=optimizer, loss='mse')

    #compute action gradients (derivative of Q values w.r.t actions)
    action_gradients = K.gradients(Q_values, actions)

    #define an additional function to fetch action gradients (to be used by actor model)
    self.get_action_gradients = K.function(
      inputs=[*self.model.input, K.learning_phase()],
      outputs=action_gradients
    )