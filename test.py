from problem import Problem
from search import LocalSearchStrategy
import math
from search import LocalSearchStrategy


def main():
    def schedule(t):
        return max(0.0, 100 * math.exp(-0.005 * t))
    filename = 'monalisa.jpg'  # Replace with the correct path to your image file
    num_trials = 10  # Number of trials for random restart

    # Create a problem instance with the given state space
    problem = Problem(filename)
    # Instantiate the local search strategy
    local_search = LocalSearchStrategy()

    # Run the random restart hill climbing algorithm to find a path
    # path = local_search.random_restart_hill_climbing(problem, num_trials)
    
    # print("Visualizing the state space...")
    # problem.show()

    # print("Visualizing the path...")
    # problem.draw_path(path)
    
    # path = local_search.simulated_annealing_search(problem, schedule)
    # print(path)
    # problem.draw_path(path)
    

    # Number of states to maintain at each step of the local beam search
    k = 5  # Example value

    # Run the local beam search algorithm to find a path
    path = local_search.local_beam_search(problem, k)

    problem.draw_path(path)

if __name__ == "__main__":
    main()
