# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import heapq

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
	print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
	
    """
    "*** YOUR CODE HERE ***"
	
	# initialise a Stack to implement DFS
    from util import Stack
    start = problem.getStartState()
    list=Stack()
    list.push((start,[]))
	
	# mark the current vertex as visited
    visited, stack = set(), [start]  
	
	# obtain next vertex to visit until stack is empty
    while list:
        vertex = list.pop()
        print vertex
        if vertex[0] in visited :
		    continue
        visited.add(vertex[0])
		
		# exit once Goal is reached and return path followed
        if problem.isGoalState(vertex[0]):
		    return vertex[1]
			
		# obtain valid successors of current vertex and put them in the stack
        next=problem.getSuccessors(vertex[0])	
        for successor, newpath ,cost in next:
			list.push((successor,vertex[1]+[newpath])) 
			
	
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
	
	# initialise a Queue to implement BFS
    from util import Queue
    start = problem.getStartState()
    list=Queue()
    list.push((start,[]))
	
	# mark the current vertex as visited
    visited, stack = set(), [start]
	
	# obtain next vertex to visit until queue is empty
    while list:
        vertex = list.pop()
        print vertex
        if vertex[0] in visited :
		    continue	
        visited.add(vertex[0])
		
		# exit once Goal is reached and return path followed
        if problem.isGoalState(vertex[0]):
		    return vertex[1]
			
		# obtain valid successors of current vertex and put them in the queue
        next=problem.getSuccessors(vertex[0])	
        for successor, newpath ,cost in next:
			list.push((successor,vertex[1]+[newpath]))

	
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
	
	# initialise a PriorityQueue to implement USC
    from util import PriorityQueue
    start = problem.getStartState()
    list=PriorityQueue()
    list.push((start,[],0),0)
	
	# mark the current vertex as visited
    visited, stack = set(), [start]
	
	# obtain the next vertex to visit based on priority until PriorityQueue is empty
    while list:
	    vertex, path, cost = list.pop()
	    if vertex in visited :
		    continue
	    visited.add(vertex)
		
		# exit once Goal is reached and return path followed
	    if problem.isGoalState(vertex):
		    return path
			
		# obtain valid successors of current vertex and put them in the PriorityQueue
	    for next_node , newpath , newcost in problem.getSuccessors(vertex):
		    if next_node not in visited:
			    total_cost= cost+ newcost   #calculate the priority of current vertex
			    list.push((next_node,path+[newpath],total_cost),total_cost)
	
	

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
	
	# initialise a PriorityQueue to implement A*
    from util import PriorityQueue
    list = PriorityQueue()
    start = problem.getStartState()
    list.push((start,[], 0), 0)
	
	# place nodes to visit in open list
    open = [(start,0)]
	
	# place nodes visited in closed list
    closed = []

	# obtain the next vertex to visit based on priority until PriorityQueue is empty
    while list:
        vertex, path, cost= list.pop()
		
		# exit once Goal is reached and return path followed
        if problem.isGoalState(vertex):
		    return path
		
		# check if next vertex found is in open list; if not update open
        if vertex not in open:
			open.append((vertex,cost))
		
		# mark  the current vertex as closed 
        if vertex not in closed:
        	closed.append(vertex)
		
	# obtain valid successors of current vertex and put them in the PriorityQueue
        	successors = problem.getSuccessors(vertex)
        	for next, newpath, ncost in successors:
        		if next not in closed:
				# calculate heuristics for current vertex
         			new_cost = cost + ncost
         			priority = new_cost + heuristic(next,problem)
         			list.push((next,path+[newpath],new_cost),priority)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
