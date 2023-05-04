# Travelling salesman problem with simulated annealing
## Introduction
### Travelling salesman problem
The Travelling Salesman Problem (TSP) is a classic optimization problem in computer science and mathematics. The problem is defined as follows: Given a set of cities and the distances between each pair of cities, the objective is to find the shortest possible route that visits each city exactly once and returns to the starting city. \
The TSP is an NP-hard problem, which means that it is computationally difficult to find the exact optimal solution for large instances of the problem. As a result, various heuristics and approximation algorithms have been developed to find good solutions within reasonable time constraints. \
The TSP has numerous real-world applications in fields such as logistics, transportation, and network design. Solving the TSP can help optimize delivery routes, minimize travel time and costs, and improve the efficiency of supply chains.

### Simulated annealing
Simulated annealing is a stochastic optimization algorithm that is commonly used to solve combinatorial optimization problems such as the Travelling Salesman Problem (TSP). The algorithm is based on the physical process of annealing, where a metal is heated and cooled to obtain a desired structure. \
In simulated annealing, the algorithm starts with an initial solution and gradually explores the solution space by making small changes to the current solution. These changes are accepted or rejected based on a probability distribution that depends on the current temperature and the change in the objective function. The algorithm then cools down over time, reducing the probability of accepting worse solutions and converging towards a good solution. \
Simulated annealing is a popular optimization algorithm due to its ability to find good solutions for complex problems with many local optima. The algorithm is flexible and can be applied to a wide range of optimization problems, including TSP, scheduling, and resource allocation. However, the performance of the algorithm depends heavily on the choice of parameters such as the cooling schedule and the initial temperature, and tuning these parameters can be a challenging task. 

