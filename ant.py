import numpy as np
import copy
from Build_up_Graph import VehicleRoutingProblemGraph
import 


class Ant:
    def __init__(self, input_graph: VehicleRoutingProblemGraph, start_index=0):
        super()
        self.start_index = start_index
        self.graph = input_graph
        self.current_index = start_index
        self.vehicle_load = 0
        self.travel_path = [start_index]

        self.index_to_visit = None

        self.total_travel_distance = 0

        # self.arrival_time = [0]
    
    def start_over(self):
        """
        - Inputs:
            None
        --------------------------------------------------------------------
        - This function is basically reseting all the values of the Ant and
        get it ready for the next iteration!
        --------------------------------------------------------------------
        -Outputs:
            None: it simply updates the values of ant to Zero point!
        """
        self.start_index = 0
        self.current_index = self.start_index
        self.vehicle_load = 0
        self.travel_path = [self.start_index]

        self.index_to_visit = None
        # self.index_to_visit.remove(start_index)

        self.total_travel_distance = 0

    def move_vehicle_to_next_node(self, next_node_index):
        """
        - Inputs:
            next_node_index: the index of the next node that we 
            want to visit! (Type: Int)
        ----------------------------------------------------------------
        - This function is moving the ant/vehicle to the next node that
        it should visit!
        ----------------------------------------------------------------
        - Outputs:
            None: it will return nothing! simply updating some parameters!
        """
        # Add the new node to the traveled path.
        self.travel_path.append(next_node_index)
        # calculate the current travel distance! & add it to the total!
        current_distance = self.graph.distance_matrix[self.current_index][next_node_index]
        self.total_travel_distance += current_distance

        if self.graph.graph_nodes[next_node_index].is_central_warehouse:
            pass
            # self.vehicle_load = 0
        else:
            # add the demand to the vehicle_load
            self.vehicle_load += self.graph.graph_nodes[next_node_index].demand

        self.current_index = next_node_index


    def can_vehicle_load_more(self, next_index) -> bool:
        """
        - Inputs:
            next_index: the index of the next node that we 
            want to visit! (Type: Int)
        ----------------------------------------------------------------
        - In this function we are checking that can we load 
        more and visit more customers or Not!
        ----------------------------------------------------------------
        -Outputs:
            result: tell's us that can we load more or not! (Type: bool)
        """
        result = None
        if self.vehicle_load + self.graph.graph_nodes[next_index].demand > self.graph.capacity_of_vehicle:
            result = False
        else: 
            result = True
        return result