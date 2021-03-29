import os
import numpy as np
import tensorflow as tf

import events as e
from .callbacks import create_model, state_to_feature_tensor
from collections import deque
from random import shuffle

# Custom events:
CLOSER_TO_CRATE = "CLOSER_TO_CRATE"
FARTHER_TO_CRATE = "FARTHER_TO_CRATE"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
FARTHER_TO_COIN = "FARTHER_TO_COIN"
CLOSER_TO_OPPONENT = "CLOSER_TO_OPPONENT"
FARTHER_TO_OPPONENT = "FARTHER_TO_OPPONENT"
REPEATING_PENALTY = "REPEATING_PENALTY"

def setup_training(self):

    # Phase I & II:
    self.PHASE_I = True # True False
    self.PHASE_II = False # True False

    # Variables for Expert:
    if self.PHASE_I:
        np.random.seed()
        self.bomb_history = deque([], 5)
        self.coordinate_history = deque([], 20)
        self.ignore_others_timer = 0
        self.current_round = 0
        self.expert_action = expert_action

    # Epsilon greedy parameters:
    self.EPSILON_MAX    = 0.5
    self.EPSILON_MIN    = 0.05
    self.EPSILON_DECAY  = 0.99999
    self.epsilon        = self.EPSILON_MAX

    # Probability for Actions in Exploration:
    self.P = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

    # Discount factor Gamma:
    self.GAMMA = 0.95

    # Learning rate Alpha:
    self.ALPHA = 0.0005
    
    # Max length of experience replay buffers:
    # Batch Size for learning:
    # Steps after model is trained:
    # Episodes after model_target is updated:
    self.HISTORY_SIZE                   = 2000*400
    self.BATCH_SIZE                     =      128
    self.STEPS_UPADTE_MODEL             =        4
    self.EPISODES_UPDATE_MODEL_TARGET   =       20
    self.STEPS_PHASE_I                  = 1000*400
    self.STEPS_PHASE_II                 = 1000*400

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
    # self.optimizer = keras.optimizers.Adam(learning_rate=self.ALPHA)
    self.model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=tf.keras.optimizers.Adam(lr=self.ALPHA),
            metrics=["accuracy"]
            )

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):

    # Test for irrelevant calls:
    if old_game_state is None:
        return
    if self_action is None:
        return
    if new_game_state is None:
        return

    # Update step_count:
    self.step_count += 1

    # Epsilon decay:
    self.epsilon *= self.EPSILON_DECAY
    self.epsilon = max(self.epsilon, self.EPSILON_MIN)

    # Calculate reward:
    custom_events = determin_custom_events(old_game_state, new_game_state)
    events.extend(custom_events)
    reward = reward_from_events(events)

    # Update episode_reward:
    self.episode_reward += reward

    # Fill Experience replay buffer:
    self.state_history.append(state_to_feature_tensor(old_game_state))
    self.action_history.append(self.ACTIONS.index(self_action))
    self.next_state_history.append(state_to_feature_tensor(new_game_state))
    self.reward_history.append(reward)
    self.done_history.append(0) # 0 = False
    
    # Train model:
    if (self.step_count % self.STEPS_UPADTE_MODEL == 0) and len(self.state_history) >= self.BATCH_SIZE:

        # Select a batch:
        indices          = np.random.choice(len(self.state_history), size=self.BATCH_SIZE, replace=False)
        state_batch      = np.array([self.state_history[i] for i in indices])
        action_batch     = np.array([self.action_history[i] for i in indices])
        next_state_batch = np.array([self.next_state_history[i] for i in indices])
        reward_batch     = np.array([self.reward_history[i] for i in indices])
        done_batch       = np.array([self.done_history[i] for i in indices])

        # Calculate Q-values according to double-Q-learning:
        Q = self.model.predict(state_batch)
        Q_next = self.model.predict(next_state_batch)
        Q_target_next = self.model_target.predict(next_state_batch)

        action_next = np.argmax(Q_next, axis=1)
        Q[range(self.BATCH_SIZE),action_batch] = reward_batch + self.GAMMA*Q_target_next[range(self.BATCH_SIZE),action_next]*(1-done_batch)

        self.model.train_on_batch(state_batch, Q)

        # Train model to batch:
        # with tf.GradientTape() as tape:
        #     Q = self.model(state_batch)
        #     Q_next = self.model.predict(next_state_batch)
        #     Q_target_next = self.model_target.predict(next_state_batch)

        #     action_next = np.argmax(Q_next, axis=1)
        #     Q_update = reward_batch + self.GAMMA*Q_target_next[range(self.BATCH_SIZE),action_next]*(1-done_batch)
        #     mask = tf.one_hot(action_next, 6)
        #     Q = tf.reduce_sum(tf.multiply(Q, masks), axis=1)
            
        #     loss = Q_update - Q

        # grads = tape.gradient(loss, self.model.trainable_variables)
        # self.optimizer.apply_gradients(zip(grads, model.trainable_variables))   

def end_of_round(self, last_game_state: dict, last_action: str, events: list):

    # Test for irrelevant calls:
    if last_game_state is None:
        return
    if last_action is None:
        return

    # Update step_count:
    self.step_count += 1
    # Update episode_count:
    self.episode_count += 1

    # Epsilon decay:
    self.epsilon *= self.EPSILON_DECAY
    self.epsilon = np.max([self.epsilon, self.EPSILON_MIN])

    # Calculate reward:
    reward = reward_from_events(events)

    # Update episode_reward_history:
    self.episode_reward += reward
    self.episode_reward_history.append(self.episode_reward)
    self.episode_reward = 0

    ###
    print("Epsiode Reward History: ", self.episode_reward_history)
    ###

    # Fill Experience replay buffer:
    self.state_history.append(state_to_feature_tensor(last_game_state))
    self.action_history.append(self.ACTIONS.index(last_action))
    self.next_state_history.append(np.zeros(126))
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
        Q = self.model.predict(state_batch)
        Q_next = self.model.predict(next_state_batch)
        Q_target_next = self.model_target.predict(next_state_batch)

        action_next = np.argmax(Q_next, axis=1)
        Q[range(self.BATCH_SIZE),action_batch] = reward_batch + self.GAMMA*Q_target_next[range(self.BATCH_SIZE),action_next]*(1-done_batch)

        # Train model to batch:
        self.model.train_on_batch(state_batch, Q)

    # Update model_target:
    if self.episode_count % self.EPISODES_UPDATE_MODEL_TARGET == 0:
        self.model_target.set_weights(self.model.get_weights())
    
    # Save model:
    self.model.save("model")

    if self.step_count >= self.STEPS_PHASE_I:
        self.PHASE_I = False
        self.PHASE_II = True
        self.epsilon = self.EPSILON_MAX/2

def determin_custom_events(old_game_state: dict, new_game_state: dict) -> list:

    custom_events = []

    position_agent_old = np.array(old_game_state["self"][3])
    position_agent_new = np.array(new_game_state["self"][3])

    positions_crates_old    = np.argwhere(old_game_state["field"] == 1)
    positions_coins_old     = np.array(old_game_state["coins"])
    positions_opponents_old = np.array([opponent[3] for opponent in old_game_state["others"]])

    if len(positions_crates_old) != 0:
        distances_crates_old = np.abs(positions_crates_old - position_agent_old).sum(1)
        distances_crates_new = np.abs(positions_crates_old - position_agent_new).sum(1)
        diff = np.min(distances_crates_old) - np.min(distances_crates_new)
        if diff > 0:
            custom_events.append(CLOSER_TO_CRATE)
        if diff < 0:
            custom_events.append(FARTHER_TO_CRATE)

    if len(positions_coins_old) != 0:
        distances_coins_old = np.abs(positions_coins_old - position_agent_old).sum(1)
        distances_coins_new = np.abs(positions_coins_old - position_agent_new).sum(1)
        diff = np.min(distances_coins_old) - np.min(distances_coins_new)
        if diff > 0:
            custom_events.append(CLOSER_TO_COIN)
        if diff < 0:
            custom_events.append(FARTHER_TO_COIN)

    if len(positions_opponents_old) != 0:
        distances_opponents_old = np.abs(positions_opponents_old - position_agent_old).sum(1)
        distances_opponents_new = np.abs(positions_opponents_old - position_agent_new).sum(1)
        diff = np.min(distances_opponents_old) - np.min(distances_opponents_new)
        if diff > 0:
            custom_events.append(CLOSER_TO_OPPONENT)
        if diff < 0:
            custom_events.append(FARTHER_TO_OPPONENT)

    if np.array_equal(state_to_feature_tensor(old_game_state),state_to_feature_tensor(new_game_state)) and np.array_equal(position_agent_old,position_agent_new):
        custom_events.append(REPEATING_PENALTY)

    return custom_events

def reward_from_events(events: list) -> int:

    # Normalization Factor:
    N = 400
    
    # Individul rewards to events:
    game_rewards = {
        e.MOVED_LEFT            : -1/N,
        e.MOVED_RIGHT           : -1/N,
        e.MOVED_UP              : -1/N,
        e.MOVED_DOWN            : -1/N,
        e.WAITED                : -2/N,
        e.INVALID_ACTION        : -5/N,

        e.BOMB_DROPPED          :  0/N,
        e.BOMB_EXPLODED         :  0/N,

        e.CRATE_DESTROYED       : +5/N,
        e.COIN_FOUND            : +7/N,
        e.COIN_COLLECTED        : +20/N,

        e.KILLED_OPPONENT       : +100/N,
        e.KILLED_SELF           : -400/N,

        e.GOT_KILLED            : -400/N,
        e.OPPONENT_ELIMINATED   :  0/N,
        e.SURVIVED_ROUND        : +100/N,

        CLOSER_TO_CRATE         :  +1/N,
        FARTHER_TO_CRATE        :  -1/N,
        CLOSER_TO_COIN          :  +1/N,
        FARTHER_TO_COIN         :  -1/N,
        CLOSER_TO_OPPONENT      :  +1/N,
        FARTHER_TO_OPPONENT     :  +1/N,
        REPEATING_PENALTY       :  -10/N,
    }
    
    # Determine reward from events:
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum

####################################################################################################
####################################################################################################
####################################################################################################

def expert_action(self, game_state: dict) -> str:
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] <= 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a

def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]
