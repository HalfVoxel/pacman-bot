# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random
import time
import util
from game import Directions
import game
import numpy as np
import math

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='DummyAgent', second='DummyAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

SIGHT_RANGE = 5

class PositionEstimator:
  ''' Estimates the position of agents '''
  def __init__(self, agent_index, gameState):
    self.index = agent_index
    self.walls = gameState.getWalls()
    self.width = self.walls.width
    self.height = self.walls.height
    self.map = np.zeros((self.width, self.height))
    initial_position = gameState.getInitialAgentPosition(self.index)
    self.map[initial_position[0],initial_position[1]] = 1.0
    self.dx = [+1, 0, -1, 0, 0]
    self.dy = [0, +1, 0, -1, 0]

  def update(self, bot, gameState):
    noisy_distance = gameState.getAgentDistances()[self.index]
    new_map = np.zeros((self.width, self.height))

    # If the target is directly observable, then just set that tile to have a probability of 1.0
    target_pos = gameState.getAgentPosition(self.index)
    if target_pos is not None:
      new_map[target_pos[0], target_pos[1]] = 1.0
      self.map = new_map
      return

    # Calculate probability of being in each tile given that the agent moves randomly
    for x in range(0, self.width):
      for y in range(0, self.height):
        if not self.walls[x][y]:
          numOptions = 0
          for i in range(len(self.dx)):
            if not self.walls[x+self.dx[i]][y+self.dy[i]]:
              numOptions += 1

          value = self.map[x,y] / numOptions
          for i in range(len(self.dx)):
            if not self.walls[x+self.dx[i]][y+self.dy[i]]:
              new_map[x+self.dx[i], y+self.dy[i]] += value

    pos = gameState.getAgentPosition(bot.index)
    for x in range(0, self.width):
      for y in range(0, self.height):
        dist = abs(pos[0]-x) + abs(pos[1]-y)
        if dist <= 5:
          # We did not observe the agent, so it cannot be on this square because then we would have observed it
          new_map[x,y] = 0
        else:
          new_map[x,y] *= gameState.getDistanceProb(dist, noisy_distance)

    # Normalize
    new_map /= np.sum(new_map)
    self.map = new_map

  def most_probable_position(self):
    return np.unravel_index(self.map.argmax(), self.map.shape)

  def draw_probabilities(self, bot, color):
    for x in range(self.width):
      for y in range(self.height):
        f = math.pow(self.map[x][y], 0.01)
        if f > 0:
          bot.debugDraw([(x, y)], color * f)


class DummyAgent(CaptureAgent):
  def __init__(self, isRed):
    super().__init__(isRed)
    self.position_estimators = []

  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    for i in range(0,4):
      self.position_estimators.append(PositionEstimator(i, gameState))

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''
    self.debugClear()
    colors = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0]])

    for pi in range(len(self.position_estimators)):
      p = self.position_estimators[pi]
      p.update(self, gameState)
      p.draw_probabilities(self, colors[pi])

    return random.choice(actions)
