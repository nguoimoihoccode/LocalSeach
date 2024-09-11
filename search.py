import random
import numpy as np
import math


class LocalSearchStrategy:
    def hill_climbing(self, problem):
        # Randomly choose an initial state (x, y)
        current_x = np.random.choice(problem.X[0])
        current_y = np.random.choice(problem.Y[:, 0])
        current_evaluation = problem.Z[current_y, current_x]
        path = [(current_x, current_y, current_evaluation)]

        while True:
            # Determine the neighbors of the current state
            neighbors = problem.get_neighbors(current_x, current_y)

            # Evaluate the neighbors and choose the best
            next_state = max(neighbors, key=lambda state: state[2])

            # Compare the best neighbor's evaluation to the current state's
            if next_state[2] <= current_evaluation:
                # If not better, we've reached the peak
                break

            # Move to the neighbor state and add to the path
            current_x, current_y, current_evaluation = next_state
            path.append(next_state)

        return path

    def random_restart_hill_climbing(self, problem, num_trial):
        best_path = []
        best_evaluation = -np.inf

        for _ in range(num_trial):
            path = self.hill_climbing(problem)
            current_evaluation = path[-1][2]  # Last state's evaluation
            if current_evaluation > best_evaluation:
                best_evaluation = current_evaluation
                best_path = path

        return best_path
    
    
    def simulated_annealing_search(self, problem, schedule):
        # Initialize the current state as a random state
        current_x = np.random.choice(problem.X[0])
        current_y = np.random.choice(problem.Y[:, 0])
        current_state = (current_x, current_y, problem.Z[current_y, current_x])
        current_path = [current_state]

        # Time variable for the schedule function
        time = 1

        while True:
            # Get the temperature from the schedule function
            temperature = schedule(time)
            if temperature <= 0:
                # If temperature is 0 or effectively 0, we stop the search
                break

            # Get neighbors and select a random neighbor as the next state
            neighbors = problem.get_neighbors(current_x, current_y)
            if not neighbors:
                break
            next_state = random.choice(neighbors)
            delta_e = np.int64(next_state[2]) - np.int64(current_state[2])
            min_temp = 1e-10
            if temperature < min_temp:
                temperature = min_temp        
            # Decide whether to move to the next state
            if delta_e > 0 or (temperature > 0 and math.exp(delta_e / temperature) > random.random()):
                # Move to the new state
                current_state = next_state
                current_path.append(current_state)
                current_x, current_y = current_state[0], current_state[1]

            # Increment time
            time += 1

        return current_path
    
    def local_beam_search(self, problem, k):
        # Randomly choose k initial states
        states = [(np.random.choice(problem.X[0]), np.random.choice(problem.Y[:, 0])) for _ in range(k)]
        states = [(x, y, problem.Z[y, x]) for x, y in states]
        
        # The path will only keep the states of one of the beams
        path = [max(states, key=lambda s: s[2])]

        while True:
            # Generate all successors for all states
            all_successors = []
            for x, y, _ in states:
                successors = problem.get_neighbors(x, y)
                all_successors.extend(successors)
                
            # Select the top k unique successors
            states = list(set(all_successors))  # Remove duplicates
            states.sort(key=lambda s: s[2], reverse=True)  # Sort based on evaluation
            states = states[:k]  # Keep only the top k
            # Update the path with the best state if it's better than the last one in the path
            best_current_state = max(states, key=lambda s: s[2])
            if best_current_state[2] > path[-1][2]:
                path.append(best_current_state)
            else:
                break

        # Return the path to the best state found
        return path

