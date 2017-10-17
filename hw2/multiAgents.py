# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        NEXT_FOOD_SCORE = 1000.0
        NEXT_GHOST_SCORE = -5000.0

        x, y = newPos
        final_score = 0.0
        # there is a food in next state
        if currentGameState.hasFood(x, y):
            final_score += NEXT_FOOD_SCORE

        # average manhanttan distance to all foods
        nxt_food_list = newFood.asList()
        tmp_score = 0.0
        for new_x, new_y in nxt_food_list:
            tmp_score += 1 / float(abs(x - new_x) + abs(y - new_y))
        nxt_food_list_size = len(nxt_food_list)
        if nxt_food_list_size != 0:
            tmp_score /= float(len(nxt_food_list))
        final_score += tmp_score

        # whether there is a ghost in the next state
        for new_x, new_y in successorGameState.getGhostPositions():
            dist = int(abs(new_x - x) + abs(new_y - y))
            if dist <= 1:
                final_score += NEXT_GHOST_SCORE
                break

        return final_score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def is_terminal_state(self, game_state, depth, agent_index):
        if game_state.isWin() or game_state.isLose() or \
           depth == self.depth and agent_index == self.num_of_agents - 1:
            return True
        return False;

    def utility(self, game_state):
        return self.evaluationFunction(game_state)

    def minimax(self, game_state, depth, agent_index):
        if self.is_terminal_state(game_state, depth, agent_index):
            return self.utility(game_state)
        nxt_agent_index, nxt_depth = agent_index + 1, depth
        if nxt_agent_index == self.num_of_agents:
            nxt_agent_index = 0
            nxt_depth += 1
        legal_moves = game_state.getLegalActions(agent_index)
        scores = []
        for action in legal_moves:
            nxt_state = game_state.generateSuccessor(agent_index, action)
            if agent_index == 0 and action == Directions.STOP:
                scores.append(-1e9)
            else:
                scores.append(self.minimax(nxt_state, nxt_depth, nxt_agent_index))
        if agent_index == 0:
            return max(scores)
        return min(scores)

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        self.num_of_agents = gameState.getNumAgents()
        legal_moves = gameState.getLegalActions()
        scores = [self.minimax(gameState.generateSuccessor(0, action), 1, 1) \
                  for action in legal_moves if action != Directions.STOP]
        best_score = max(scores)
        best_indices = [idx for idx in range(len(scores)) if scores[idx] == best_score]
        chosen_index = random.choice(best_indices)
        return legal_moves[chosen_index]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def is_terminal_state(self, game_state, depth, agent_index):
        if game_state.isWin() or game_state.isLose() or \
           depth == self.depth and agent_index == self.num_of_agents - 1:
            return True
        return False;

    def utility(self, game_state):
        return self.evaluationFunction(game_state)

    def alphabeta(self, game_state, depth, agent_index, alpha, beta):
        if self.is_terminal_state(game_state, depth, agent_index):
            return self.utility(game_state)
        nxt_agent_index, nxt_depth = agent_index + 1, depth
        if nxt_agent_index == self.num_of_agents:
            nxt_agent_index = 0
            nxt_depth += 1
        legal_moves = game_state.getLegalActions(agent_index)
        if agent_index == 0:
            v = -1e9
            for action in legal_moves:
                nxt_state = game_state.generateSuccessor(agent_index, action)
                v = max([v, self.alphabeta(nxt_state, nxt_depth, nxt_agent_index, alpha, beta)])
                if v >= beta:
                    return v;
                alpha = max([alpha, v])
            return v
        else:
            v = 1e9
            for action in legal_moves:
                nxt_state = game_state.generateSuccessor(agent_index, action)
                v = min([v, self.alphabeta(nxt_state, nxt_depth, nxt_agent_index, alpha, beta)])
                if alpha >= v:
                    return v;
                beta = min([beta, v])
            return v

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        self.num_of_agents = gameState.getNumAgents()
        legal_moves = gameState.getLegalActions()
        scores = [self.alphabeta(gameState.generateSuccessor(0, action), 1, 1, -1e9, 1e9) \
                  for action in legal_moves if action != Directions.STOP]
        best_score = max(scores)
        best_indices = [idx for idx in range(len(scores)) if scores[idx] == best_score]
        chosen_index = random.choice(best_indices)
        return legal_moves[chosen_index]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

