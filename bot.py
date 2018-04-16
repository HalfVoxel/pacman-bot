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
from game import Directions, Actions
import game
import numpy as np
import math
import collections

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
SCARED_TIME = 40

dx = [+1, 0, -1, 0, 0]
dy = [0, +1, 0, -1, 0]
dd = [Directions.EAST, Directions.NORTH, Directions.WEST, Directions.SOUTH, Directions.STOP]


class PositionEstimator:
  ''' Estimates the position of agents '''

  def __init__(self, agent_index, gameState):
    self.index = agent_index
    self.walls = gameState.getWalls()
    self.width = self.walls.width
    self.height = self.walls.height
    self.scaredTimer = 0

    # Map of probabilities of where the agent is given the observations and that it moves sort of randomly.
    # Perhaps most important is the fact that this will be non-zero for precisely the tiles that the agent could be at.
    self.map = np.zeros((self.width, self.height))
    initial_position = gameState.getInitialAgentPosition(self.index)
    self.map[initial_position[0], initial_position[1]] = 1.0

  def update(self, bot, gameState):
    self.scaredTimer -= 1
    # If the target is directly observable, then just set that tile to have a probability of 1.0
    target_pos = gameState.getAgentPosition(self.index)
    if target_pos is not None:
      self.known_position(target_pos)
      return

    noisy_distance = gameState.getAgentDistances()[self.index]
    new_map = np.zeros((self.width, self.height))

    # Calculate probability of being in each tile given that the agent moves randomly
    for x in range(0, self.width):
      for y in range(0, self.height):
        if not self.walls[x][y]:
          numOptions = 0
          for i in range(len(dx)):
            if not self.walls[x + dx[i]][y + dy[i]]:
              numOptions += 1

          value = self.map[x, y] / numOptions
          for i in range(len(dx)):
            if not self.walls[x + dx[i]][y + dy[i]]:
              new_map[x + dx[i], y + dy[i]] += value

    pos = gameState.getAgentPosition(bot.index)
    for x in range(0, self.width):
      for y in range(0, self.height):
        dist = abs(pos[0] - x) + abs(pos[1] - y)
        if dist <= SIGHT_RANGE:
          # We did not observe the agent, so it cannot be on this square because then we would have observed it
          new_map[x, y] = 0
        else:
          new_map[x, y] *= gameState.getDistanceProb(dist, noisy_distance)

    # Normalize
    new_map /= np.sum(new_map)
    self.map = new_map

  def most_probable_position(self):
    return np.unravel_index(self.map.argmax(), self.map.shape)

  def scare(self):
    self.scaredTimer = SCARED_TIME - 1

  def can_be_at(self, p):
    return self.map[p[0], p[1]] > 0

  def known_position(self, p):
    self.map = np.zeros((self.width, self.height))
    self.map[p[0], p[1]] = 1.0

  def draw_probabilities(self, bot, color):
    #m = self.reachable_in_time_map()
    #m *= 0.01
    m = self.map
    for x in range(self.width):
      for y in range(self.height):
        f = min(1.0, math.pow(m[x][y], 0.01))
        if f > 0:
          bot.debugDraw([(x, y)], color * f)

  def reachable_in_time_map(self, account_for_scared):
    ''' Minimum number of ticks it takes for the agent to reach each tile on the map given the observations '''
    temp = np.zeros(self.map.shape)
    que = collections.deque()
    for x in range(self.width):
      for y in range(self.height):
        if self.map[x, y] > 0:
          temp[x, y] = 0
          que.append((x, y))
        else:
          temp[x, y] = math.inf

    # Do a BFS
    while(len(que) > 0):
      x, y = que.popleft()

      v = temp[x, y] + 1
      for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if not self.walls[nx][ny]:
          if v < temp[nx, ny]:
            temp[nx, ny] = v
            que.append((nx, ny))

    if account_for_scared:
      temp = np.maximum(temp, self.scaredTimer)

    return temp


class DummyAgent(CaptureAgent):
  def __init__(self, isRed):
    super().__init__(isRed)
    self.position_estimators = []
    self.tick = -1

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

    for i in range(0, 4):
      self.position_estimators.append(PositionEstimator(i, gameState))

  def search(self, gameState, start, ends, time_to_reach, opponents, clearence):
    temp = np.ones(time_to_reach[0].shape) * 100000
    width = time_to_reach[0].shape[0]
    height = time_to_reach[0].shape[1]
    parents = [[None for y in range(height)] for x in range(width)]
    temp[start[0], start[1]] = 0
    que = collections.deque()
    que.append(start)
    walls = gameState.getWalls()

    # Do a BFS
    while(len(que) > 0):
      x, y = que.popleft()
      if (x, y) in ends:
        p = (x,y)
        if p == start:
          return Directions.STOP, 0

        length = 0
        while parents[p[0]][p[1]] != start:
          p = parents[p[0]][p[1]]
          length += 1

        return Actions.vectorToDirection((p[0] - start[0], p[1] - start[1])), length

      v = temp[x, y] + 1
      for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if not walls[nx][ny]:
          clear = 100000
          if (nx < width//2) != self.red or self.scared:
            # On opponent side
            for opp in opponents:
              clear = min(clear, time_to_reach[opp][nx,ny] - v)

          if clear >= clearence and v < temp[nx, ny]:
            temp[nx, ny] = v
            parents[nx][ny] = (x,y)
            que.append((nx, ny))

    return None, None

  def border_nodes(self):
    walls = self.getCurrentObservation().getWalls()
    width = walls.width
    height = walls.height
    border_coord = (width//2 + (-1 if self.red else 0))
    return [(border_coord, y) for y in range(height) if not walls[border_coord][y]]

  def find_path_to_safe_region(self, gameState, start, must_visit, time_to_reach, opponents, min_clearance, require_return=True):
    ''' Returns best clearence to enemies '''

    # Do a binary search over the minimum clearence to enemies
    # A negative value indicates that the enemy might catch us
    mn = min_clearance
    mx = 20

    border_nodes = self.border_nodes()

    if must_visit is not None:
      for opp in opponents:
        mx = min(mx, time_to_reach[opp][must_visit[0], must_visit[1]] - self.getMazeDistance(start, must_visit) + 1)

    best_dir = None
    while mn + 1 < mx:
      mid = (mn + mx) // 2

      if must_visit is not None:
        dir1, length1 = self.search(gameState, start, [must_visit], time_to_reach, opponents, mid)
        if require_return:
          dir2, length2 = self.search(gameState, must_visit, border_nodes, time_to_reach, opponents, mid + length1)
        else:
          # Whatever
          dir2 = Directions.NORTH
          length2 = 0

        if dir1 is None or dir2 is None:
          mx = mid
        else:
          mn = mid
          best_dir = dir1
      else:
        dir2, length2 = self.search(gameState, start, border_nodes, time_to_reach, opponents, mid)
        if dir2 is None:
          mx = mid
        else:
          mn = mid
          best_dir = dir2

    # print("Best path has a clearence of " + str(mn) + " " + str(best_dir))
    return mn, best_dir

  def update_food(self):
    obs = self.getCurrentObservation()
    last = self.getPreviousObservation()
    if last is not None:
      currentFood = self.getFoodYouAreDefending(obs)
      changed = np.array(self.getFoodYouAreDefending(last).data) - np.array(currentFood.data)
      for x in range(currentFood.width):
        for y in range(currentFood.height):
          if changed[x,y]:
            num = 0
            for p in self.position_estimators:
              if p.can_be_at((x,y)):
                num += 1

            if num == 1:
              for p in self.position_estimators:
                p.known_position((x,y))
            elif num == 0:
              print("??? Food was eaten, but no agent could be there")

      if len(self.getCapsules(obs)) != len(self.getCapsules(last)):
        # Just ate a capsule to scare enemies
        # TODO: Might have off by one errors in timing
        for p in self.position_estimators:
          p.scare()

  def chooseAction(self, gameState):
    self.tick += 1

    """
    Picks among actions randomly.
    """
    state = gameState.getAgentState(self.index)
    self.scared = state.scaredTimer > 0

    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''
    self.debugClear()
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])

    for pi in range(len(self.position_estimators)):
      p = self.position_estimators[pi]
      p.update(self, gameState)

    self.update_food()

    for pi in range(len(self.position_estimators)):
      p = self.position_estimators[pi]
      if pi == 1:
        p.draw_probabilities(self, colors[pi])


    # Find the dot wich has the highest score of (distance to enemy - distance from us)
    # Find the closest dot which we can guarantee to be able to escape to our side from
    food = self.getFood(gameState)
    opponents = self.getOpponents(gameState)
    time_to_reach = [p.reachable_in_time_map(True) for p in self.position_estimators]
    position = gameState.getAgentPosition(self.index)
    best_dir = None
    best_score = -100
    for x in range(food.width):
      for y in range(food.height):
        if food[x][y]:
          for opp in opponents:
            clearence, dir = self.find_path_to_safe_region(gameState, position, (x, y), time_to_reach, opponents, best_score)
            if clearence > best_score:
              best_score = clearence
              best_dir = dir

    # Always grab capsules if we can reach them before the enemy can
    capsules = self.getCapsules(gameState)
    for capule in capsules:
      clearence, dir = self.find_path_to_safe_region(gameState, position, capule, time_to_reach, opponents, 0, False)
      if clearence > 0:
        best_score = clearence
        best_dir = dir

    if state.numCarrying > 0:
      clearence, dir = self.find_path_to_safe_region(gameState, position, None, time_to_reach, opponents, best_score)
      if clearence > best_score and best_score < 0:
        best_score = clearence
        best_dir = dir

    if not state.isPacman and best_score < 0:
      dir, length = self.search(gameState, position, self.border_nodes(), time_to_reach, opponents, -100)
      print("Moving to border " + str(self.tick) + " " + str(length))
      assert dir is not None
      best_dir = dir
    else:
      print("Moving with " + str(self.tick) + ": " + str(best_score))

    if best_dir is not None:
      return best_dir
    else:
      return Directions.STOP
