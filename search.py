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
from game import Directions
from typing import List
import random
import numpy as np


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

def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def euclideanDistance(position, problem):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    #util.raiseNotDefined()
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # Usando uma fila
    states2explore = util.Queue()      
  
    startState = problem.getStartState()

    print("Start:", problem.getStartState())
    print("Goal:", problem.getGoalState())
    
    
    startNode = (startState, [], 0) #(state, action, cost)

    exploredStates = []
    
    # O estado inicial eh armazenado nos estados que serao explorados
    states2explore.push(startNode)
    
    while not states2explore.isEmpty():
        # Recupera o proximo estado
        currentState, actions, currentCost = states2explore.pop()
        
        if currentState not in exploredStates:
            #put popped node state into explored list
            exploredStates.append(currentState)
            #print("Is the current state goal?", problem.isGoalState(currentState))
            #print("Euclidean: ", euclideanDistance(currentState, problem))
            if problem.isGoalState(currentState):
                return actions
            else:
                #list of (successor, action, stepCost)
                #print("Current successors:", problem.getSuccessors(currentState))
                successors = problem.getSuccessors(currentState)
                
                for succState, succAction, succCost in successors:
                    newAction = actions + [succAction]
                    newCost = currentCost + succCost
                    newNode = (succState, newAction, newCost)

                    states2explore.push(newNode)

    return actions

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Usando uma fila
    states2explore = util.Stack()      
    
    startState = problem.getStartState()
    startNode = (startState, [], 0) #(state, action, cost)

    exploredStates = []
    
    # O estado inicial eh armazenado nos estados que serao explorados
    states2explore.push(startNode)
    
    while not states2explore.isEmpty():
        # Recupera o proximo estado
        currentState, actions, currentCost = states2explore.pop()
        
        if currentState not in exploredStates:
            #put popped node state into explored list
            exploredStates.append(currentState)

            if problem.isGoalState(currentState):
                return actions
            else:
                #list of (successor, action, stepCost)
                successors = problem.getSuccessors(currentState)
                
                for succState, succAction, succCost in successors:
                    newAction = actions + [succAction]
                    newCost = currentCost + succCost
                    newNode = (succState, newAction, newCost)

                    states2explore.push(newNode)

    return actions

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Usando uma fila
    states2explore = util.PriorityQueue()      
    
    startState = problem.getStartState()
    startNode = (startState, [], 0) #(state, action, cost)

    exploredStates = {}
    
    # O estado inicial eh armazenado nos estados que serao explorados
    states2explore.push(startNode, 0)
    
    while not states2explore.isEmpty():
        #begin exploring first (lowest-cost) node on states2explore
        currentState, actions, currentCost = states2explore.pop()
       
        if (currentState not in exploredStates) or (currentCost < exploredStates[currentState]):
            #put popped node's state into explored list
            exploredStates[currentState] = currentCost

            if problem.isGoalState(currentState):
                return actions
            else:
                #list of (successor, action, stepCost)
                successors = problem.getSuccessors(currentState)
               
                for succState, succAction, succCost in successors:
                    newAction = actions + [succAction]
                    newCost = currentCost + succCost
                    newNode = (succState, newAction, newCost)

                    states2explore.update(newNode, newCost)

    return actions


def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def greedySearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    nextState = None
    
    startState = problem.getStartState()
    currentState = startState

    actions = []
    exploredStates = []
    currentCost = 0

    import sys
    
    while currentState != None:

        if problem.isGoalState(currentState) or currentState == None:
            return actions
        else:
            exploredStates.append(currentState)
            successors = problem.getSuccessors(currentState)

            if len(problem.getSuccessors(currentState)) == 0 and not problem.isGoalState(currentState):
                return actions
            
            bestAction = []
            bestCost = 0
            bestSucc = None
            currentHeuristic = sys.maxsize

            for succState, succAction, succCost in successors:
                if succState not in exploredStates:
                    distTemp = euclideanDistance(succState, problem)
                    if distTemp < currentHeuristic:
                        bestAction = [succAction]
                        bestCost = succCost
                        bestSucc = succState
                        currentHeuristic = distTemp

            actions = actions + bestAction
            currentCost = bestCost + currentCost            
            #currentState = (bestSucc, actions, currentCost)
            #print("Current:", currentState)
            #print("Next:", bestSucc)
            currentState = bestSucc

    #print(actions)
    return actions

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    states2explore = util.PriorityQueue()

    exploredNodes = [] #holds (state, cost)

    startState = problem.getStartState()
    startNode = (startState, [], 0) #(state, action, cost)

    states2explore.push(startNode, 0)

    while not states2explore.isEmpty():

        #begin exploring first (lowest-combined (cost+heuristic) ) node on states2explore
        currentState, actions, currentCost = states2explore.pop()

        #put popped node into explored list
        exploredNodes.append((currentState, currentCost))

        if problem.isGoalState(currentState):
            return actions

        else:
            #list of (successor, action, stepCost)
            successors = problem.getSuccessors(currentState)

            #examine each successor
            for succState, succAction, succCost in successors:
                newAction = actions + [succAction]
                newCost = problem.getCostOfActions(newAction)
                newNode = (succState, newAction, newCost)

                #check if this successor has been explored
                already_explored = False
                for explored in exploredNodes:
                    #examine each explored node tuple
                    exploredState, exploredCost = explored

                    if (succState == exploredState) and (newCost >= exploredCost):
                        already_explored = True

                #if this successor not explored, put on states2explore and explored list
                if not already_explored:
                    states2explore.push(newNode, newCost + heuristic(succState, problem))
                    exploredNodes.append((succState, newCost))

    return actions

# Execute com o comando python3 pacman.py -l mediumMaze -p SearchAgent -a fn=ga   
def geneticAlgorithm(problema: SearchProblem):
    # Parâmetros do algoritmo
    geracoes = 10  # Número máximo de gerações
    taxa_mutacao = 0.2  # Probabilidade de ocorrer mutação em um cromossomo
    tamanho_populacao = 100  # Quantidade de indivíduos na população
    solucao = None  # Melhor solução encontrada
    custo_global = float('inf')  # Custo da melhor solução
    taxa_convergencia = int(geracoes / 3)  # Número de gerações sem melhoria para convergência
    geracoes_sem_ganho = 0  # Contador de gerações sem melhoria

    # Função auxiliar para obter um índice aleatório
    def obterIndiceAleatorio(tamanho_maximo):
        return random.randint(0, tamanho_maximo - 1)

    # Gera um cromossomo aleatório, que é uma sequência de ações
    def gerarCromossomoAleatorio():
        cromossomo = []  # Representação do cromossomo
        estadosExplorados = []  # Estados já visitados para evitar loops
        estadoAtual = problema.getStartState()  # Estado inicial do problema
        estadosExplorados.append(estadoAtual)

        # Gera uma sequência de ações aleatórias até atingir um limite
        for _ in range(150):  # Limite de 150 ações
            sucessores = problema.getSuccessors(estadoAtual)  # Sucessores do estado atual
            # Seleciona um sucessor aleatório que ainda não foi explorado
            estadoSucessor, acaoSucessor, custoSucessor = sucessores[obterIndiceAleatorio(len(sucessores))]
            if estadoSucessor not in estadosExplorados:
                cromossomo.append(acaoSucessor)  # Adiciona a ação ao cromossomo
                estadoAtual = estadoSucessor  # Atualiza o estado atual
                estadosExplorados.append(estadoSucessor)  # Marca o estado como explorado

        return cromossomo

    # Avalia o custo de um cromossomo
    def avaliar(cromossomo):
        estado = problema.getStartState()  # Começa do estado inicial
        custo = 0  # Custo acumulado

        # Percorre as ações no cromossomo
        for acao in cromossomo:
            sucessores = problema.getSuccessors(estado)  # Sucessores do estado atual
            acoesValidas = {s[1]: (s[0], s[2]) for s in sucessores}  # Mapeia ações válidas para seus resultados

            if acao not in acoesValidas:  # Se a ação não é válida, retorna custo infinito
                return float('inf')

            estado, custoPasso = acoesValidas[acao]  # Atualiza o estado e acumula o custo
            custo += custoPasso

        # Verifica se o estado final é o estado objetivo
        if not problema.isGoalState(estado):
            return float('inf')

        return custo

    # Realiza o cruzamento entre dois pais para gerar um novo cromossomo (filho)
    def cruzamento(pai1, pai2):
        pontoCorte = random.randint(1, min(len(pai1), len(pai2)) - 1)  # Define um ponto de corte
        return pai1[:pontoCorte] + pai2[pontoCorte:]  # Combina partes dos pais

    # Aplica mutação em um cromossomo
    def mutacao(cromossomo):
        if cromossomo:
            indiceMutacao = random.randint(0, len(cromossomo) - 1)  # Escolhe um índice para mutação
            estado = problema.getStartState()  # Estado inicial
            # Executa ações até o índice de mutação
            for acao in cromossomo[:indiceMutacao]:
                sucessores = problema.getSuccessors(estado)
                acoesValidas = {s[1]: s[0] for s in sucessores}
                if acao in acoesValidas:
                    estado = acoesValidas[acao]
                else:
                    return cromossomo
            
            sucessores = problema.getSuccessors(estado)  # Obtém sucessores do estado atual
            if sucessores:  # Seleciona uma nova ação aleatória
                cromossomo[indiceMutacao] = random.choice(sucessores)[1]
        return cromossomo

    # Inicializa a população com cromossomos aleatórios
    populacao = [gerarCromossomoAleatorio() for _ in range(tamanho_populacao)]

    # Itera ao longo de um número fixo de gerações
    for _ in range(geracoes):
        # Ordena a população com base nos custos dos cromossomos
        populacao = sorted(populacao, key=lambda x: avaliar(x))

        # Obtém o melhor cromossomo da geração
        melhorCromossomo = populacao[0]
        custo_local = avaliar(melhorCromossomo)
        if custo_local < custo_global:  # Atualiza a melhor solução encontrada
            solucao = melhorCromossomo
            custo_global = custo_local
            geracoes_sem_ganho = 0  # Reseta o contador de gerações sem ganho
        else:
            geracoes_sem_ganho += 1  # Incrementa o contador

        # Para o algoritmo se não houver melhoria por várias gerações
        if geracoes_sem_ganho >= taxa_convergencia:
            return solucao

        # Seleciona os melhores indivíduos (metade superior)
        selecionados = populacao[:tamanho_populacao // 2]
        filhos = []

        # Realiza cruzamento até completar a nova população
        while len(filhos) < tamanho_populacao - len(selecionados):
            pai1, pai2 = random.sample(selecionados, 2)  # Seleciona dois pais aleatoriamente
            filho = cruzamento(pai1, pai2)  # Gera um filho
            filhos.append(filho)

        # Aplica mutação nos filhos
        for filho in filhos:
            if random.random() < taxa_mutacao:  # Decide se ocorrerá mutação com base na taxa
                mutacao(filho)

        # Atualiza a população para a próxima geração
        populacao = selecionados + filhos

    # Retorna o melhor cromossomo encontrado
    return min(populacao, key=lambda x: avaliar(x))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
ucs = uniformCostSearch
gdy = greedySearch
astar = aStarSearch

ga = geneticAlgorithm