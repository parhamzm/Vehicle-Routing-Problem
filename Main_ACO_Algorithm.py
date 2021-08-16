import numpy as np
import random
from Build_up_Graph_Map import VehicleRoutingProblemDrawGraph
from Build_up_Graph import VehicleRoutingProblemGraph, PathMessage
from ant import Ant
from threading import Thread
from queue import Queue
import time
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="talk", style="darkgrid", palette="hls", font="sans-serif", font_scale=1.05)


class VRPAntColonyOptimization:
    def __init__(self, input_graph: VehicleRoutingProblemGraph, number_of_ants=10, 
        max_iter=200, beta=5, q0=0.8, alpha=2, show_map_or_plot_results=True):
        """
        - Inputs:
            input_graph: the builded graph from file (Type: VehicleRoutingProblemGraph)
            number_of_ants: The number of ants that we want to use!
            max_iter = Maximum iterations that we will use!
            beta: Beta Parameter!
            q0: (exploitation / exploration) parameter
            alpha: alpha Parameter
            show_map: Wheter or not draw Results!
        --------------------------------------------------------------
        - This funtion basically initializes parameters of the class!
        --------------------------------------------------------------
        - Outputs:
            Nothing
        """
        super()
        # Set the Input Graph:
        self.graph = input_graph
        # Set the Number of Ants:
        self.number_of_ants = number_of_ants
        self.optimize_number_of_ants()
        # max_iter : Maximum Iteration
        self.max_iter = max_iter
        # vehicle_capacity : 
        self.max_load = input_graph.capacity_of_vehicle
        # beta Parameter:
        self.beta = beta
        # alpha Parameter:
        self.alpha = alpha
        # q0 Parameter: (exploitation / exploration)
        self.q0 = q0
        # best path
        self.best_path_distance = math.inf
        # Saves the best achieved path so far!
        self.best_path = None
        self.best_vehicle_num = None
        self.best_result_each_iteration = None
        # Initializating Ants according to the given Number!
        self.ants = list(Ant(self.graph) for _ in range(self.number_of_ants))

        self.show_map = show_map_or_plot_results

    def optimize_number_of_ants(self):
        """
        - Inputs:
            Nothing
        -----------------------------------------------------------
        - This function is basically optimizing the number_of_ants
        based on the number of vehicles.
        -----------------------------------------------------------
        -Outputs:

        """
        if (self.number_of_ants % self.graph.number_of_vehicle) != 0:
            self.number_of_ants -= (self.number_of_ants % self.graph.number_of_vehicle)
    
    def plot_leaning_fitness(self):
        """
        -------------------
        This function will draw for us the fitness plot.
        -------------------
        """
        plt.figure(figsize=(16, 10))
        plt.plot([i for i in range(len(self.best_result_each_iteration))], 
                            self.best_result_each_iteration, c='dodgerblue')
        best_res = np.argmin(self.best_result_each_iteration)
        worst_res = np.argmax(self.best_result_each_iteration)
        line_init = plt.axhline(y=self.best_result_each_iteration[worst_res], color='r', linestyle='--')
        plt.text(-0.5, best_res, best_res)#, **text_style)
        line_min = plt.axhline(y=self.best_result_each_iteration[best_res], color='g', linestyle='--')
        plt.text(-0.5, worst_res, worst_res)#, **text_style)
        plt.legend([line_init, line_min], ['Worst Case', 'Best Case'])
        plt.ylabel('Traveled Distance')
        plt.xlabel('Iteration')
        plt.title("Best Result:=> {}".format(self.best_path_distance))
        plt.show()

    def ant_colony_optimization_start_algorithm(self):
        """
        - Inputs:
            Nothing
        -------------------------------------------------------
        -
        -------------------------------------------------------
        -Outputs:
            Nothing!
        """
        # Start Ant Colony Opt Algorithm:
        path_queue_for_figure = Queue()
        aco_thread = Thread(target=self.ant_colony_optimization, args=(path_queue_for_figure,))
        aco_thread.start()

        # Show Figure or Not:
        if self.show_map:
            figure = VehicleRoutingProblemDrawGraph(self.graph, path_queue_for_figure)
            figure.run()
        aco_thread.join()

        # 
        if self.show_map:
            path_queue_for_figure.put(PathMessage(None, None))


    def ant_colony_optimization(self, path_queue_for_figure: Queue):
        """
        - Inputs:
            
        -------------------------------------------------------
        - This function runs The Ant Colony Optimization Algorithm!
        -------------------------------------------------------
        -Outputs:

        """
        start_time_total = time.time()
        # print("Number of Ants: ", self.number_of_ants)
        batch_number = int(self.number_of_ants / self.graph.number_of_vehicle)
        # Set the Maximum Iteration Counter
        start_iteration = 0
        each_iteration_best_distasnce = []
        each_iteration_best_path = []
        for my_iter in range(self.max_iter):
            ant_counter = 0
            iteration_paths_distances = []
            for batch in range(batch_number):
                # Iterate through the Ants for Solution!
                remaining_nodes_to_visit = list(range(0, self.graph.number_of_nodes))
                remaining_nodes_to_visit.remove(self.graph.warehouse_index)
                batch_paths_distances = []
                for k in range(self.graph.number_of_vehicle):
                    # print("K: ", k)
                    # Go through All the Customers with Ants:
                    while len(remaining_nodes_to_visit) != 0:
                        # Select the Next Node (Customer) to visit!
                        next_index = self.select_next_index(self.ants[ant_counter], remaining_nodes_to_visit)

                        if not self.ants[ant_counter].can_vehicle_load_more(next_index):
                            next_index = self.select_next_index(self.ants[ant_counter], remaining_nodes_to_visit)
                            if not self.ants[ant_counter].can_vehicle_load_more(next_index):
                                next_index = self.select_next_index(self.ants[ant_counter], remaining_nodes_to_visit)
                                if not self.ants[ant_counter].can_vehicle_load_more(next_index):
                                    next_index = self.graph.warehouse_index
                        if next_index is not self.graph.warehouse_index:
                            remaining_nodes_to_visit.remove(next_index)
                            # print("Remaining to Visit: ", remaining_nodes_to_visit)
                        # Move the Ant to the next Node:
                        self.ants[ant_counter].move_vehicle_to_next_node(next_index)
                        # self.graph.local_update_pheromone(self.ants[ant_counter].current_index, next_index)
                        if self.graph.graph_nodes[next_index].is_central_warehouse:
                            if self.ants[ant_counter].total_travel_distance != 0:
                                batch_paths_distances.append(self.ants[ant_counter].total_travel_distance)
                            break
                    if len(remaining_nodes_to_visit) == 0:
                        self.ants[ant_counter].move_vehicle_to_next_node(0)
                    ant_counter+=1
                batch_paths_distances = np.array(batch_paths_distances)
                batch_distance = np.sum((batch_paths_distances))
                iteration_paths_distances.append(batch_distance)
            best_index_in_iteration = np.argmin(iteration_paths_distances)
            each_iteration_best_distasnce.append(iteration_paths_distances[best_index_in_iteration])
            each_iteration_best_path.append(self.ants[int(best_index_in_iteration)].travel_path)
            if iteration_paths_distances[best_index_in_iteration] < self.best_path_distance:
                best_idx_start = best_index_in_iteration * self.graph.number_of_vehicle
                best_path = []
                for i in range(best_idx_start, best_idx_start+ self.graph.number_of_vehicle):
                    best_path.append(self.ants[i].travel_path)
                self.best_path = best_path
                self.best_path_distance = iteration_paths_distances[best_index_in_iteration]
                self.best_vehicle_num = self.graph.number_of_vehicle
                start_iteration = my_iter
                print('\n')
                print('[iteration %d]: find a improved path, its distance is %f' % (my_iter, self.best_path_distance))
                print('it takes %0.3f second multiple_ant_colony_system running' % (time.time() - start_time_total))
            # self.graph.global_update_pheromone(self.best_path, self.best_path_distance)
            for ant in self.ants:
                self.graph.global_update_pheromone(ant.travel_path, ant.total_travel_distance)
                ant.start_over()
            self.graph.pheremone_evaporation()
            print("==============================")
            print("| iteration:==> ", my_iter)
        print('\n')
        print("|================================================================|")
        print("| Final Best Path distance :===> {0} |".format(self.best_path_distance))
        print("| Number of Vehicles:===> {0} |".format(self.best_vehicle_num))
        print("|================================================================|")
        print("|==================>>> 'Best Path Solution' <<<==================|")
        for ant_count, ant_path in enumerate(self.best_path):
            print("| Vehicle '{0}' path:=> {1} |".format(ant_count+1, ant_path))
        print("|================================================================|")

        print('| It takes %0.3f second running |' % (time.time() - start_time_total))
        print("|================================================================|")
        self.best_result_each_iteration = each_iteration_best_distasnce
        # Show Path in the Map:
        if self.show_map:
            path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))
        
        if not self.show_map:
            self.plot_leaning_fitness()

    def select_next_index(self, ant: Ant, remaining_nodes=[]):
        """
        - Inputs:
            ant: The ant that want's to select next Node.
            remaining_nodes: nodes that are remaining to select!
        -----------------------------------------------------------
        - This function will select the best next node for us
        -----------------------------------------------------------
        - Outputs:
            next_node = we will return the next node that we will visit!
        """
        current_index = ant.current_index
        transition_prob = np.power(self.graph.virtual_pheromone_matrix[current_index][remaining_nodes], self.alpha) * \
            np.power(self.graph.distance_matrix[current_index][remaining_nodes], -1 * self.beta)

        transition_prob = transition_prob / np.sum(transition_prob)

        if np.random.rand() < self.q0:
            max_prob_index = np.argmax(transition_prob)
            next_index = remaining_nodes[max_prob_index]
        else:
            # Now we will randomly select one of the remaining nodes!
            next_index = VRPAntColonyOptimization.stochastic_accept(remaining_nodes, transition_prob)
        return next_index

    @staticmethod
    def stochastic_accept(index_to_visit, transition_prob): # Randomly select the next Vertex
        """
        - Inputs:
            index_to_visit:
            transition_prob:
        -------------------------------------------------------
        - in this Function we will choose next node randomly!
        -------------------------------------------------------
        -Outputs:

        """
        # calculate N and max fitness value
        N = len(index_to_visit)

        # normalize
        sum_tran_prob = np.sum(transition_prob)
        norm_transition_prob = transition_prob / sum_tran_prob

        # select: O(1)
        while True:
            # randomly select an individual with uniform probability
            index = int(N * random.random())
            if random.random() <= norm_transition_prob[index]:
                return index_to_visit[index]



if __name__ == '__main__':
    file_path = 'data.txt'
    ants_num = 100
    beta = 2
    alpha = 0.5
    q0 = 0.1
    show_figure = True
    max_iteration = 300
    graph = VehicleRoutingProblemGraph(file_path)
    macs = VRPAntColonyOptimization(input_graph=graph, max_iter=max_iteration, number_of_ants=ants_num, 
                                    beta=beta, alpha=alpha, q0=q0, show_map_or_plot_results=show_figure)
    macs.ant_colony_optimization_start_algorithm()