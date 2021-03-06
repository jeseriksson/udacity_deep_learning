
from keras import layers, models, optimizers
from keras import backend as K
import numpy as np

class Actor:
  #actor (policy) model

  def __init__(self, state_size, action_size, action_low, action_high):
    #initialise parameters and build model

    self.state_size = state_size
    self.action_size = action_size
    self.action_low = action_low
    self.action_high = action_high
    self.action_range = self.action_high - self.action_low

    #initialise other variabels
    self.learning_rate = 0.0001

    self.build_model()
  
  def build_model(self):
    #build an actor (policy) network that maps states to actions

    #define input layer (states)
    states = layers.Input(shape=(self.state_size,), name='states')

    #add hidden layers

    net = layers.Dense(units=400, kernel_regularizer=layers.regularizers.l2(1e-6))(states)
    net = layers.BatchNormalization()(net)
    net = layers.Activation('relu')(net)

    net = layers.Dense(units=300, kernel_regularizer=layers.regularizers.l2(1e-6))(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation('relu')(net)

    #add final output layer with weights initialised to Uniform[-3e-3, 3e-3]
    raw_actions = layers.Dense(units=self.action_size, activation='sigmoid', kernel_initializer = layers.initializers.RandomUniform(minval=-0))

    #scale [0, 1] output for each action dimension to proper range
    actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)

    #create keras model
    self.model = models.Model(inputs=states, outputs=actions)

    #define loss function using action value (Q value) gradients
    action_gradients = layers.Input(shape=(self.action_size,))
    loss = K.mean(-action_gradients * actions)

    #incorporate any additional losses

    #define optimizer and training function
    optimizer = optimizers.Adam(lr=self.learning_rate)
    updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
    self.train_fn = K.function(
      inputs=[self.model.input, action_gradients, K.learning_phase()],
      outputs=[],
      updates=updates_op
    )

