import atari_py
import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import time

def show_state(env, step=0, info=""):
    env.render()

class Agent:
    def __init__(self):
        self.env     = gym.make("Breakout-v4")
        self.state   = self.env.reset()
        self.rewards = []
        self.steps   = []
        self.poilcy  = []

    def select_action(self, state, action):
        return action

    def play_episode(self, env):
        total_reward = 0.0
        state   = env.reset()
        action  = random.choice([0,1,2,3])
        steps   = 0
        actions = []
        while True:
            self.env.render()
            action = self.select_action(env, state, action, 0)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            steps += 1
            actions.append(action)
            if is_done:
                break
            state = new_state
        return total_reward, steps, actions

    def repeat(self,n_sample):
        rewards = []
        steps   = []
        policy  = []
        for i in range(n_sample):
            reward, step, actions = self.play_episode(self.env)
            rewards.append(reward)
            steps.append(step)
            policy.append(actions)
        self.rewards = rewards
        self.steps   = steps
        self.policy  = np.array(policy)


""" My best jab at a simple reflex agent that plays the game by observing the
    RGB array that represents the game. Searches for the ball whenever it is under
    the red bar and over the paddle, and moves the paddle to position it under
    the ball
"""
class ReflexAgent(Agent):
    """ Given a game state, determines the x coordinate of ball or paddle
    """
    def findCoord(self, entity, state):
        xCoord = 80 # default x coordinate is center of screen

        if (entity == 'ball'): # look for the ball
            for x in range(8, 152): # the portion of the screen between the two walls
                for y in range(70, 188): # portion between lower 3rd of screen and the paddle
                    pix = state[y][x] # pix contains pixel data
                    if pix[0]==200 and pix[1]==72 and pix[2]==72: # first red pixel found
                        xCoord = x
        elif (entity == 'paddle'): # look for the paddle
            for x in range(8, 152): # the portion of the screen between the two walls
                pix = state[190][x] # contains pixel data on the paddle's y value
                if pix[0]==200 and pix[1]==72 and pix[2]==72: # first red pixel found
                    xCoord = x
        return xCoord

    """ Given ball and paddle x coordinates, returns direction paddle should move
    """
    def choose_direction(self, ballX, paddleX):
        if (ballX < paddleX-8): # ball is to left of paddle,
            return 3            # so move left
        elif (ballX > paddleX): # ball is to the right of paddle,
            return 2            # so move right
        else:
            return 0            # otherwise, do nothing

    """ Selects the action like a simple reflex agent, given RGB array of game
        Highest score achieved: 175, Average Score: ~55
    """
    def select_action(self, env, state, action, depth):

        ballX   = self.findCoord('ball',   state) # find ball x coordinate
        paddleX = self.findCoord('paddle', state) # find paddle x coordinate

        if ballX == 80: # play 1 when ball cannot be found (ball is too high on)
            return 1
        else: # for any
            return self.choose_direction(ballX, paddleX)

""" Combines the reactivity of a reflex agent with the predictivity of the minimax.
The algorithm observes the game to determine the direction the ball is moving.
If the ball is moving down and is within ~40 pixels of the paddle, then the agent
generates a minimax tree of maximum depth n. Higher depths terminate prematurely,
and I haven't found a workaround for this yet.
"""
class MinimaxReflexAgent(Agent):
    """ Given a game state, determines the y coordinate of the top left corner of the ball
    """
    def findBallY(self, state):
        yCoord = -1
        for x in range(8, 152): # the portion of the screen between the two walls
            for y in range(160, 188): # portion between lower 3rd of screen and the paddle
                pix = state[y][x]
                if pix[0]==200 and pix[1]==72 and pix[2]==72: # first red pixel found
                    yCoord = y
        return yCoord

    """ Given Atari Breakout env and an action, determines the difference in the y coordinates of the
    ball before and after the action, returning whether or not the ball is moving downward
    """
    def isBallMovingDown(self, env, oldState, action):
        cloneState = env.clone_state()
        newState = env.step(action)[0]
        env.restore_state(cloneState)

        ballImminent = False
        if self.findBallY(oldState)!=-1 and self.findBallY(newState)!=-1:
            ballImminent = True

        if (ballImminent and self.findBallY(oldState) < self.findBallY(newState)):
            return True
        return False

    """ Selects the action for a minimax/reflex agent. If the breakout ball is moving downward and
    is in the lower third of the screen, agent generates a small minimax tree thats deep enough
    to determine if the ball will miss the paddle (if it worked)
    """
    def select_action(self, env, state, action, depth):
        maxDepth  = 5 # stores max depth of the minimax tree
        bestMove  = 1 # stores the best move found by minimax
        bestScore = 0 # stores the best score found by minimax
        tempScore = 0 # stores temporary scores for comparison to best

        # if the ball is moving down and is in the lower quarter of the screen, generate a minimac tree
        if (self.isBallMovingDown(env, state, action)):

            for move in range(4): # create a minimax branch for each possible move
                state = env.clone_state() # save current state so minimax tree does not overwrite it
                newState, tempScore, isDone, _ = env.step(move) #
                if isDone or depth >= maxDepth: # stop recursion and restore environment
                    env.restore_state(state)    # if max depth has been reached or if
                    break                       # episode is complete
                tempScore = self.select_action(env, newState, move, depth+1) # create a tree branch
                env.restore_state(state) # return state to its cloned value

                if (tempScore > bestScore): # if the tempScore is better than the best score so far,
                    bestScore = tempScore   # update the best score to that of the temp
                    bestMove  = move        # and store the new best move to be returned later
                elif (tempScore == bestScore):                 # if best and temp are equal,
                    bestMove = random.choice([bestMove, move]) # choose between them randomly

            if depth == 0:      # Return the best move if we have searched the
                return bestMove # entire minimax tree
            else:                # otherwise,
                return bestScore # return the best score for further comparison

        else:                               # if the ball is moving away from the paddle,
            return random.choice([0,1,2,3]) # choose a random move.

""" Replay the best policy found. I do not use it, considering my agents are
purely reactionary and the results are not replicable
"""
def replay(policy):
    env  = gym.make("Breakout-v4")
    obs  = env.reset()
    step = 0
    for a in policy:
        env.step(a)
        time.sleep(0.01)
        show_state(env, step)
        step+=1

reflexA = ReflexAgent()
reflexA.repeat(5)
rewards1 = reflexA.rewards
plt.hist(rewards1,bins=50)
print(np.mean(rewards1), np.std(rewards1), max(rewards1))
best_reflex = reflexA.policy[np.argmax(rewards2)]
replay(best_reflex)

""" Uncomment this to see my attempt at writing a "miniflex" agent. It sort of works,
    but it plays horribly, slowly, and terminates too soon... It doesn't work.
miniflexA = MinimaxReflexAgent()
miniflexA.repeat(5)
rewards2 = miniflexA.rewards
plt.hist(rewards2,bins=50)
print(np.mean(rewards2), np.std(rewards2), max(rewards2))
"""

""" Completely removed any attempt at implementing this agent. It was useless.
minimaxA = MinimaxAgent()
minimaxA.repeat(5)
rewards3 = minimaxA.rewards
plt.hist(rewards3,bins=50)
print(np.mean(rewards3), np.std(rewards3), max(rewards3))
"""














""""""
