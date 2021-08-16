import copy
import numpy as np



class Vertex:
    def __init__(self, index:  int, x_coordinate: float, y_coordinate: float, demand: float):
        """
        - Inputs:
            index: the id or index of the input Vertex
            x_coordinate: the x axis coordinate of the vertex
            y_coordinate: the y axis coordinate of the vertex
            demand: the suplies that customer in this vertex needs!
        ---------------------------------------------------------------
        - This is the initialization function for Vertex class!
        ---------------------------------------------------------------
        - Outpurs:
            RETURNS: None
            - Just update parameters
        """
        super()
        self.index = index

        if demand == 0:
            self.is_central_warehouse = True
            self.warehouse_index = index
        else:
            self.is_central_warehouse = False
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.demand = demand


class VehicleRoutingProblemGraph:
    def __init__(self, input_file_path, Rho=0.1):
        '''
        -Inputs:
            - input_file_path: The file that we want to read info from!
            - Rho: Rho value! used for virtual pheremone evaporation!
        ----------------------------------------------------------------------------------
        - This initialization function basically sets the init value for out graph!
        ----------------------------------------------------------------------------------
        - Outputs:
            None! : only sets the init values!
        '''
        super()
        self.number_of_nodes = 0
        self.graph_nodes = None
        self.number_of_vehicle = 0
        self.capacity_of_vehicle = 0
        # this function reads info from file and sets the init values from it!
        self.initialize_parameters_from_file(input_file_path=input_file_path)

        self.distance_matrix = None
        # This function sets the distance matrix!
        self.calculate_distance_matrix()
        # rho : Pheromone volatilization rate
        self.Rho = Rho

        # Pheromone Matrix:
        self.init_pheromone_mat = np.ones((self.number_of_nodes, self.number_of_nodes))
        self.init_pheromone_mat = 1/(self.init_pheromone_mat * self.number_of_nodes)

        self.virtual_pheromone_matrix = np.ones((self.number_of_nodes, 
                                self.number_of_nodes)) * self.init_pheromone_mat
        # Set the index of the warehouse:
        self.warehouse_index = 0
        self.get_warehouse_index()


    def copy(self, init_pheromone_val=None):
        new_graph = copy.deepcopy(self)
        if not None:
            new_graph.init_pheromone_val = init_pheromone_mat
            new_graph.virtual_pheromone_matrix = np.ones((new_graph.number_of_nodes, 
                                new_graph.number_of_nodes)) * init_pheromone_mat
        return new_graph

    def initialize_parameters_from_file(self, input_file_path="data.txt"):
        """
        - Inputs:
            input_file_path: the path of the input file. (Type: String)
        ---------------------------------------------------
        - Outputs:
            number_of_vertex: number of the vertices or nodes that the graph has.

        """
        graph_nodes_list = []
        with open(input_file_path, 'rt') as input_file:
            line_number = 1
            for line in input_file:
                if line_number == 1:
                    number_of_vehicle, capacity_of_vehicle = line.split()
                    number_of_vehicle = int(number_of_vehicle)
                    capacity_of_vehicle = int(capacity_of_vehicle)
                    print("Vehicle Num: ", number_of_vehicle)
                    print("Vehicle Capacity: ", capacity_of_vehicle)
                elif line_number >= 2:
                    graph_nodes_list.append(line.split())
                line_number += 1
        number_of_nodes = len(graph_nodes_list)
        for node in graph_nodes_list:
            print("Item: ", node)
        graph_nodes = list(Vertex(int(vertex_item[0]), float(vertex_item[1]), 
                float(vertex_item[2]), float(vertex_item[3])) for vertex_item in graph_nodes_list)

        self.number_of_nodes = number_of_nodes
        self.graph_nodes = copy.deepcopy(graph_nodes)
        self.number_of_vehicle = number_of_vehicle
        self.capacity_of_vehicle = capacity_of_vehicle


    def calculate_distance_matrix(self):
        '''
        - Inputs:
            None: this function basically uses the values 
                from the class (with using Self)
        ----------------------------------------------------------------
        - This function is creating a matrix which will represent the 
        "Euclidean Distance" between the nodes of the Graph!
        ----------------------------------------------------------------
        - Outputs:
            None: it will simply update the values of the class!
        '''
        distance_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes))
        for i in range(self.number_of_nodes):
            node_a = self.graph_nodes[i]
            distance_matrix[i][i] = 1e-9
            for j in range(i+1, self.number_of_nodes):
                node_b = self.graph_nodes[j]
                distance_matrix[i][j] = VehicleRoutingProblemGraph.calculate_euclidean_distance(node_a, node_b)
                distance_matrix[j][i] = distance_matrix[i][j]
        # print("Hello: ", distance_matrix)
        self.distance_matrix = copy.deepcopy(distance_matrix)
        

    @staticmethod
    def calculate_euclidean_distance(point_A, point_B):
        '''
        - Inputs:
            - point_A: the first point coordinates! (Type: Vertex class)
            - point_B: the second point coordinates! (Type: Vertex class)
        -----------------------------------------------------------------------------
        - This function is going to calculate the "Euclidean Distance" between
        two given points as parameters of this function!
        -----------------------------------------------------------------------------
        - Outputs:
            - euclidean_distance: the calculated "Euclidean Distance" between
            the two input point! (Type: Float)
        '''
        euclidean_distance = np.linalg.norm((point_A.x_coordinate - point_B.x_coordinate, 
                                            point_A.y_coordinate - point_B.y_coordinate))
        return euclidean_distance

    def local_update_pheromone(self, start_ind, end_ind):
        # distance = self.distance_matrix[start_ind][end_ind]
        # self.virtual_pheromone_matrix[start_ind][end_ind] += 1 / distance
        self.virtual_pheromone_matrix[start_ind][end_ind] = (1-self.Rho) * self.virtual_pheromone_matrix[start_ind][end_ind] + \
                                                  self.Rho * self.init_pheromone_val
    
    def pheremone_evaporation(self):
        """
        - Inputs:
            None
        -------------------------------------------------------------------
        - This function is basically eveporating some portion of the pheremone!
        -------------------------------------------------------------------
        - Outputs:
            Nothing! : this function only updates the pheremone Matrix!
        """
        self.virtual_pheromone_matrix = (1-self.Rho) * self.virtual_pheromone_matrix


    def global_update_pheromone(self, traveled_path, path_distance):
        """
        - Inputs:
        -------------------------------------------------------------------
        - This function is basically updating the Pheremone Matrix!
        -------------------------------------------------------------------
        - Outputs:
            Nothing! : this function only updates the pheremone Matrix!
        """
        current_index = traveled_path[0]
        for next_index in traveled_path[1:]:
            self.virtual_pheromone_matrix[current_index][next_index] += 1 / path_distance
            current_index = next_index

    def get_warehouse_index(self):
        """
        - Inputs:
            None
        -------------------------------------------------------------------
        - This function is basically updating the Pheremone Matrix!
        -------------------------------------------------------------------
        - Outputs:
            Nothing! : this function only finds and sets the index of the warehouse!
        """
        idx = None
        for node in self.graph_nodes:
            if node.is_central_warehouse is True:
                idx = node.index
        if idx is None:
            print("No index found!")
        else:
            # print("Index Found: ", idx)
            self.warehouse_index = idx -1


class PathMessage:
    def __init__(self, path, distance):
        if path is not None:
            self.path = copy.deepcopy(path)
            self.distance = copy.deepcopy(distance)
            self.used_vehicle_num = self.path.count(0) - 1
        else:
            self.path = None
            self.distance = None
            self.used_vehicle_num = None

    def get_path_info(self):
        return self.path, self.distance, self.used_vehicle_num