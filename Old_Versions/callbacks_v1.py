import os
import numpy as np
import tensorflow as tf

def setup(self):

    # Action space:
    self.ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

    # Test if Keras model exist:
    if not os.path.exists("model"):
        # Create Keras model:
        create_model().save("model")

    # Load existing Keras model:    
    self.model = tf.keras.models.load_model("model")

def act(self, game_state: dict) -> str:

    # Exploration vs. Exploitation in training:
    if self.train and np.random.uniform() < self.epsilon:
        # Choose random action in training with probability epsilon:
        self.logger.debug("Choosing action purely at random.")
        action = np.random.choice(self.ACTIONS)
        return action
    else:
        # Use the action the model predicts:
        self.logger.debug("Querying model for action.")
        feature_tensor = state_to_features(game_state)
        prediction = self.model.predict(feature_tensor[np.newaxis,:,:,:])
        action = self.ACTIONS[np.argmax(prediction)]
        return action

def create_model() -> tf.keras.Sequential:
    #   Keras Convolutional Neuronal Network
    #       Input: np.array with shape (17,17,6)
    #       Output: np.array with shape (6)
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu", input_shape=(17,17,6)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu"),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(6, activation="linear")
            ])
    return model

def state_to_features(game_state: dict) -> np.array:
    
    # Temporary variables:
    tmp_field  = game_state["field"]
    tmp_coins  = game_state["coins"]
    tmp_bombs  = game_state["bombs"]
    tmp_others = game_state["others"]
    tmp_self   = game_state["self"]
    
    # Create crate map:
    crate_map = np.zeros(tmp_field.shape)
    crate_map[tmp_field==1] = 1 
    # Create coin map:
    tmp = np.array([[coin[0],coin[1]] for coin in tmp_coins]).T
    coin_map = np.zeros(tmp_field.shape)
    if tmp.size != 0:
        coin_map[tmp[0],tmp[1]] = 1
    # Create bomb map:
    tmp = np.array([[bomb[0][0],bomb[0][1],bomb[1]+1] for bomb in tmp_bombs]).T
    bomb_map = np.zeros(tmp_field.shape)
    if tmp.size != 0:
        bomb_map[tmp[0],tmp[1]] = tmp[2]
    # Create player + player_atack map:
    tmp = [[player[3][0],player[3][1],-1,player[2]] for player in tmp_others]
    tmp.append([tmp_self[3][0],tmp_self[3][1],1,tmp_self[2]])
    tmp = np.array(tmp).T
    player_map = np.zeros(tmp_field.shape)
    player_map[tmp[0],tmp[1]] = tmp[2]
    player_atack_map = np.zeros(tmp_field.shape)
    player_atack_map[tmp[0],tmp[1]] = tmp[3]
    # Create obstacle map:
    obstacle_map = np.zeros(tmp_field.shape)
    obstacle_map[tmp_field!=0] = 1
    obstacle_map[bomb_map!=0] = 1
    obstacle_map[player_map==-1] = 1

    # Combine features together into one feature tensor:
    feature_list = [obstacle_map, crate_map, bomb_map, coin_map, player_map, player_atack_map]
    feature_tensor = np.stack(feature_list, axis=2)
    return feature_tensor
