import matplotlib.pyplot as plt
from Build_up_Graph import VehicleRoutingProblemGraph
from multiprocessing import Queue as MPQueue
import seaborn as sns
sns.set(context="talk", style="darkgrid", palette="hls", font="sans-serif", font_scale=1.05)


class VehicleRoutingProblemDrawGraph:
    def __init__(self, my_graph: VehicleRoutingProblemGraph, path_queue: MPQueue):
        """
        matplotlib drawing calculation is placed on the main thread

        """
        self.my_graph = my_graph
        self.nodes = my_graph.graph_nodes
        self.warehouse_index = my_graph.warehouse_index
        self.figure = plt.figure(figsize=(10, 10))
        self.figure_ax = self.figure.add_subplot(1, 1, 1)
        self.path_queue = path_queue
        self.warehouse_color = 'darkblue'
        self._customer_color = 'crimson'
        self._line_color = 'darksalmon'
        self.line_color_list = ['lime', 'gold', 
            'deepskyblue', 'orangered', 'magenta', 'blueviolet', 
            'royalblue', 'lawngreen', 'indigo', 'deeppink',
            'darkturquoise', 'springgreen', 'aquamarine', 'darkorange',
            'mediumslateblue', 'aqua']

    def draw_graph_points(self):
        print("Warehouse Index: ", self.warehouse_index)
        self.figure_ax.scatter([self.nodes[self.warehouse_index].x_coordinate], 
                                        [self.nodes[self.warehouse_index].y_coordinate], 
                                        c=self.warehouse_color, label='Central Warehouse', s=50)

        self.figure_ax.scatter(list(node.x_coordinate for node in self.nodes[1:]),
                               list(node.y_coordinate for node in self.nodes[1:]), 
                               c=self._customer_color, label='Customer', s=20)
        plt.pause(0.5)


    def run(self):
        # Draw Graph Nodes:
        self.draw_graph_points()
        # Show the figure:
        self.figure.show()

        while True:
            if not self.path_queue.empty():
                info = self.path_queue.get()
                while not self.path_queue.empty():
                    info = self.path_queue.get()

                path, distance, used_vehicle_num = info.get_path_info()
                if path is None:
                    print('[draw figure]: exit')
                    break

                remove_obj = []
                for line in self.figure_ax.lines:
                    if line._label == 'line':
                        remove_obj.append(line)

                for line in remove_obj:
                    self.figure_ax.lines.remove(line)
                remove_obj.clear()

                self.figure_ax.set_title('Travel Distance:=> %0.2f, || Number of Vehicles:=> %d ' % (distance, self.my_graph.number_of_vehicle))
                self.draw_graph_lines(path)
            plt.pause(1)

    def draw_graph_lines(self, paths):
        """
        """
        print("Path: ", paths)
        for path, line_color in zip(paths, self.line_color_list):
            for i in range(1, len(path)):
                x_list = [self.nodes[path[i - 1]].x_coordinate, self.nodes[path[i]].x_coordinate]
                y_list = [self.nodes[path[i - 1]].y_coordinate, self.nodes[path[i]].y_coordinate]
                self.figure_ax.plot(x_list, y_list, color=line_color, linewidth=2, label='line')
                plt.pause(0.2)