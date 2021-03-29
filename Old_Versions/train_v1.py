import os
import numpy as np
import tensorflow as tf

import events as e
from .callbacks import create_model, state_to_features
from collections import deque

# Custom events:
CUSTOM_EVENT = "PLACEHOLDER"

def setup_training(self):

    # Epsilon greedy parameters:
    self.EPSILON_MAX    = 1.0
    self.EPSILON_MIN    = 0.01
    self.EPSILON_DECAY  = 0.999
    self.epsilon        = self.EPSILON_MAX

    # Discount factor Gamma:
    self.GAMMA = 0.99

    # Learning rate Alpha:
    self.ALPHA = 0.00025
    
    # Max length of experience replay buffers:
    # Batch Size for learning:
    # Steps after model is trained:
    # Episodes after model_target is updated:
    self.HISTORY_SIZE                   = 4000
    self.BATCH_SIZE                     =   32
    self.STEPS_UPADTE_MODEL             =    4
    self.EPISODES_UPDATE_MODEL_TARGET   =   10

    # Experience replay buffers:
    self.state_history      = deque(maxlen=self.HISTORY_SIZE)
    self.action_history     = deque(maxlen=self.HISTORY_SIZE)
    self.next_state_history = deque(maxlen=self.HISTORY_SIZE)
    self.reward_history     = deque(maxlen=self.HISTORY_SIZE)
    self.done_history       = deque(maxlen=self.HISTORY_SIZE)

    # Reward parameters:
    self.episode_reward = 0
    self.episode_reward_history = []
    self.step_count = 0
    self.episode_count = 0
      
    # model_target:
    self.model_target = tf.keras.models.load_model("model")
    
    # Compile model:
    self.model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=tf.keras.optimizers.Adam(lr=self.ALPHA),
            metrics=["accuracy"]
            )

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):

    # Test for first call:
    if old_game_state is None:
        return

    # Update step_count:
    self.step_count += 1

    # Epsilon decay:
    self.epsilon *= self.EPSILON_DECAY
    self.epsilon = np.max([self.epsilon, self.EPSILON_MIN])

    # Calculate reward:
    reward = reward_from_events(self, events)

    # Update episode_reward:
    self.episode_reward += reward

    # Fill Experience replay buffer:
    self.state_history.append(state_to_features(old_game_state))
    self.action_history.append(self.ACTIONS.index(self_action))
    self.next_state_history.append(state_to_features(new_game_state))
    self.reward_history.append(reward)
    self.done_history.append(0) # 0=False
    
    # Train model:
    if (self.step_count % self.STEPS_UPADTE_MODEL == 0) and len(self.state_history) >= self.BATCH_SIZE:

        # Select a batch:
        indices          = np.random.choice(len(self.state_history), size=self.BATCH_SIZE, replace=False)
        state_batch      = np.array([self.state_history[i] for i in indices])
        action_batch     = np.array([self.action_history[i] for i in indices])
        next_state_batch = np.array([self.next_state_history[i] for i in indices])
        reward_batch     = np.array([self.reward_history[i] for i in indices])
        done_batch       = np.array([self.done_history[i] for i in indices])

        # Calculate Q-values to batch:
        Q_batch = self.model.predict(state_batch)
        Q_next = self.model_target.predict(next_state_batch)
        Q_target = reward_batch + self.GAMMA * np.max(Q_next, axis=1) * (1-done_batch)
        Q_batch[np.arange(self.BATCH_SIZE),action_batch] = Q_target

        # Train model to batch:
        self.model.train_on_batch(state_batch, Q_batch)
            
def end_of_round(self, last_game_state: dict, last_action: str, events: list):

    # Update step_count:
    self.step_count += 1
    # Update episode_count:
    self.episode_count += 1

    # Epsilon decay:
    self.epsilon *= self.EPSILON_DECAY
    self.epsilon = np.max([self.epsilon, self.EPSILON_MIN])

    # Calculate reward:
    reward = reward_from_events(self, events)

    # Update episode_reward_history:
    self.episode_reward += reward
    self.episode_reward_history.append(self.episode_reward)
    self.episode_reward = 0

    ###
    print("Epsiode Reward History: ", self.episode_reward_history)
    ###

    # Fill Experience replay buffer:
    self.state_history.append(state_to_features(last_game_state))
    self.action_history.append(self.ACTIONS.index(last_action))
    self.next_state_history.append(np.zeros((17,17,6)))
    self.reward_history.append(reward)
    self.done_history.append(1) # 1=True

    # Train model:
    if (self.step_count % self.STEPS_UPADTE_MODEL == 0) and len(self.state_history) >= self.BATCH_SIZE:

        # Select a batch:
        indices          = np.random.choice(len(self.state_history), size=self.BATCH_SIZE, replace=False)
        state_batch      = np.array([self.state_history[i] for i in indices])
        action_batch     = np.array([self.action_history[i] for i in indices])
        next_state_batch = np.array([self.next_state_history[i] for i in indices])
        reward_batch     = np.array([self.reward_history[i] for i in indices])
        done_batch       = np.array([self.done_history[i] for i in indices])

        # Calculate Q-values to batch:
        Q_batch = self.model.predict(state_batch)
        Q_next = self.model_target.predict(next_state_batch)
        Q_target = reward_batch + self.GAMMA * np.max(Q_next, axis=1) * (1-done_batch)
        Q_batch[np.arange(self.BATCH_SIZE),action_batch] = Q_target

        # Train model to batch:
        self.model.train_on_batch(state_batch, Q_batch)

    # Update model_target:
    if self.episode_count % self.EPISODES_UPDATE_MODEL_TARGET == 0:
        self.model_target.set_weights(self.model.get_weights())
    
    # Save model:
    self.model.save("model")

def reward_from_events(self, events: list) -> int:
    
    # Individul rewards to events:
    game_rewards = {
        e.MOVED_LEFT            : -1,
        e.MOVED_RIGHT           : -1,
        e.MOVED_UP              : -1,
        e.MOVED_DOWN            : -1,
        e.WAITED                : -1,
        e.INVALID_ACTION        : -5,

        e.BOMB_DROPPED          : -1,
        e.BOMB_EXPLODED         :  0,

        e.CRATE_DESTROYED       : +5,
        e.COIN_FOUND            : +7,
        e.COIN_COLLECTED        : +11,

        e.KILLED_OPPONENT       : +9,
        e.KILLED_SELF           : -9,

        e.GOT_KILLED            : -9,
        e.OPPONENT_ELIMINATED   :  0,
        e.SURVIVED_ROUND        : +5,

        CUSTOM_EVENT            :  0,
    }
    
    # Determine reward from events:
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum
