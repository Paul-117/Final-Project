import os
import numpy as np
import tensorflow as tf

def setup(self):

    # Action space:
    self.ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

    # Test if Keras model exist:
    if not os.path.exists("model"):
        # Otherwise create Keras model:
        create_model().save("model")

    # Load existing Keras model:    
    self.model = tf.keras.models.load_model("model")

def act(self, game_state: dict) -> str:

    # Training:
    if self.train:
        # Phase I: Learning from Expert:
        if self.PHASE_I:        
            # Exploration with probability epsilon:
            if np.random.uniform() < self.epsilon:
                self.logger.debug("Phase I: Expert-Learning Exploration")
                action = np.random.choice(self.ACTIONS, p=self.P)
            # Exploitation:
            else:
                self.logger.debug("Phase I: Expert-Learning Exploitation")
                action = self.expert_action(self, game_state)
        # Phase II: Self-Learning with Exploitation and Exploration:
        if self.PHASE_II:
            # Exploration with probability epsilon:
            if np.random.uniform() < self.epsilon:
                self.logger.debug("Phase II: Self-Learning Exploration")
                action = np.random.choice(self.ACTIONS, p=self.P)
            # Exploitation:
            else:
                self.logger.debug("Phase II: Self-Learning Exploitation")
                feature_tensor = state_to_feature_tensor(game_state)
                Q_prediction = self.model.predict(feature_tensor[np.newaxis,:])
                action = self.ACTIONS[np.argmax(Q_prediction)]
        # Return Action:
        return action 

    # Playing:
    self.logger.debug("Playing")
    feature_tensor = state_to_feature_tensor(game_state)
    Q_prediction = self.model.predict(feature_tensor[np.newaxis,:])
    action = self.ACTIONS[np.argmax(Q_prediction)]
    return action

def create_model() -> tf.keras.Sequential:
    #   Keras Deep Neuronal Network
    #       Input: np.array with shape (126)
    #       Output: np.array with shape (6)
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(90, activation="relu", input_dim=126),
            tf.keras.layers.Dense(6, activation="linear")
            ])
    return model

def state_to_feature_tensor(game_state: dict) -> np.array:
    
    # Temporary variables:
    tmp_field  = game_state["field"]
    tmp_coins  = game_state["coins"]
    tmp_bombs  = game_state["bombs"]
    tmp_others = game_state["others"]
    tmp_self   = game_state["self"]

    dim_x ,dim_y = tmp_field.shape
    pad_x = 1
    pad_y = 1
    
    # Create maps:
    ############################################################
    # obstacle_map: 0 = free        ; 1     = none-free
    # crate_map:    0 = no-crate    ; 1     = crate
    # coin_map:     0 = no-coin     ; 1     = coin
    # bomb_map:     0 = no-bomb     ; 1..4  = bomb-timer
    # opponent_map: 0 = no-opponent ; 1     = opponent    
    ############################################################
    obstacle_map = np.zeros((dim_x,dim_y))
    crate_map    = np.zeros((dim_x,dim_y))
    coin_map     = np.zeros((dim_x,dim_y))
    bomb_map     = np.zeros((dim_x,dim_y))
    opponent_map = np.zeros((dim_x,dim_y))
    
    # Create variable:
    bomb_possible = tmp_self[2]

    # Fill maps:
    # Obstacle map:
    obstacle_map[tmp_field!=0] = 1
    # Crate map:
    crate_map[tmp_field == 1] = 1
    # Coin map:
    for coin in tmp_coins:
        coin_map[coin] = 1
    # Bomb map:
    for bomb in tmp_bombs:
        obstacle_map[bomb[0]] = 1
        x_bomb = bomb[0][0]
        y_bomb = bomb[0][1]
        x_min_bomb = max(1, x_bomb-3) if y_bomb%2==1 else x_bomb
        x_max_bomb = min(15, x_bomb+3) if y_bomb%2==1 else x_bomb
        y_min_bomb = max(1, y_bomb-3) if x_bomb%2==1 else y_bomb
        y_max_bomb = min(15, y_bomb+3) if x_bomb%2==1 else y_bomb
        bomb_map[x_min_bomb:x_max_bomb+1,y_bomb] = bomb[1]+1
        bomb_map[x_bomb,y_min_bomb:y_max_bomb+1] = bomb[1]+1
    # Opponent map:
    for opponent in tmp_others:
        obstacle_map[opponent[3]] = 1
        opponent_map[opponent[3]] = 1

    # Add padding:
    obstacle_map = np.pad(obstacle_map, [(pad_x, pad_x), (pad_y, pad_y)], mode='constant', constant_values=1)
    crate_map    = np.pad(crate_map, [(pad_x, pad_x), (pad_y, pad_y)], mode='constant', constant_values=0)
    coin_map     = np.pad(coin_map, [(pad_x, pad_x), (pad_y, pad_y)], mode='constant', constant_values=0)
    bomb_map     = np.pad(bomb_map, [(pad_x, pad_x), (pad_y, pad_y)], mode='constant', constant_values=0)
    opponent_map = np.pad(opponent_map, [(pad_x, pad_x), (pad_y, pad_y)], mode='constant', constant_values=0)
    
    # Local version of maps:
    x_self, y_self = tmp_self[3]
    x_self += 1
    y_self += 1

    obstacle_map_local = obstacle_map[x_self-2:x_self+3,y_self-2:y_self+3]
    crate_map_local = crate_map[x_self-2:x_self+3,y_self-2:y_self+3]
    coin_map_local = coin_map[x_self-2:x_self+3,y_self-2:y_self+3]
    bomb_map_local = bomb_map[x_self-2:x_self+3,y_self-2:y_self+3]
    opponent_map_local = opponent_map[x_self-2:x_self+3,y_self-2:y_self+3]

    # Combine features together into one feature tensor:
    feature_list = [obstacle_map_local, crate_map_local, coin_map_local, bomb_map_local, opponent_map_local]
    feature_tensor = np.stack(feature_list, axis=2)
    ###
    #print_features(feature_tensor, bomb_possible)
    ###
    feature_tensor = feature_tensor.flatten()
    feature_tensor = np.append(feature_tensor,bomb_possible)
    return feature_tensor

def print_features(feature_tensor, bomb_possible):
    print("obstacle_map_local:")
    print(feature_tensor[:,:,0].T)
    print("crate_map_local:")
    print(feature_tensor[:,:,1].T)
    print("coin_map_local:")
    print(feature_tensor[:,:,2].T)
    print("bomb_map_local:")
    print(feature_tensor[:,:,3].T)
    print("opponent_map_local:")
    print(feature_tensor[:,:,4].T)
    print("bomb_possible:")
    print(bomb_possible)
