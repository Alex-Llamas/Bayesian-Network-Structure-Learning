from collections import defaultdict
import xml.etree.cElementTree as ElementTree
import pandas as pd
# from Plots import *
from DataSet import *
import binSelection
import matplotlib.pyplot as plt
import CorrelationMesures as CorreMeasures
import time


# ------- Class for a directed graph and functions to find cycles ---------
class Graph:
    def __init__(self, df):
        # No. of vertices
        self.dataFrame = df
        self.category_threshold = 54
        self.V = df.columns.__len__()  # number of vertices
        self.nodeNames = df.columns
        self.Edges = 0
        # default dictionary to store graph
        self.graph = defaultdict(list)
        self.components = defaultdict(list)
        self.componentIndex = 0
        self.Time = 0
        self.plot = False
        self.dependencies = np.zeros((self.V, self.V))  # matrix with the dependencies

    # function to add an edge to graph
    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.Edges += 1

    # used to find cycles
    def scc_util(self, u, low, disc, stack_member, st):

        # Initialize discovery time and low value
        disc[u] = self.Time
        low[u] = self.Time
        self.Time += 1
        stack_member[u] = True
        st.append(u)

        # Go through all vertices adjacent to this
        for v in self.graph[u]:

            # If v is not visited yet, then recur for it
            if disc[v] == -1:

                self.scc_util(v, low, disc, stack_member, st)

                # Check if the subtree rooted with v has a connection to
                # one of the ancestors of u
                # Case 1 (per above discussion on Disc and Low value)
                low[u] = min(low[u], low[v])

            elif stack_member[v]:
                '''Update low value of 'u' only if 'v' is still in stack
                (i.e. it's a back edge, not cross edge).
                Case 2 (per above discussion on Disc and Low value) '''
                low[u] = min(low[u], disc[v])

        # head node found, pop the stack and print an scc
        w = -1  # To store stack extracted vertices
        if low[u] == disc[u]:
            while w != u:
                w = st.pop()
                self.components[self.componentIndex].append(w)
                # print w,
                stack_member[w] = False
            self.componentIndex += 1
            # print""

    # Used to find cycles. The function to do DFS traversal. It uses recursive scc_util()
    def scc(self):

        # Mark all the vertices as not visited
        # and Initialize parent and visited,
        # and ap(articulation point) arrays
        disc = [-1] * self.V
        low = [-1] * self.V
        stack_member = [False] * self.V
        st = []

        # Call the recursive helper function
        # to find articulation points
        # in DFS tree rooted with vertex 'i'
        for i in range(self.V):
            if disc[i] == -1:
                self.scc_util(i, low, disc, stack_member, st)

    def to_xml(self, network_name="test.bayes"):
        df = self.dataFrame
        network = ElementTree.Element("Network", {"version": "7.2",
                                                  "name": "example",
                                                  "nodeCount": str(self.V),
                                                  "linkCount": str(self.Edges),
                                                  "variableCount": str(self.V)})

        nodes = ElementTree.SubElement(network, "Nodes", {"count": str(self.V)})
        # node = ElementTree.SubElement(nodes, "Node", {"id":"0"})
        node = defaultdict(list)
        node_distribution_options = defaultdict(list)
        variable = defaultdict(list)
        states = defaultdict(list)
        for idx in range(self.V):
            node[idx] = ElementTree.SubElement(nodes, "Node", {"id": str(idx),
                                                               "variableCount": "1",
                                                               "name": self.nodeNames[idx],
                                                               "x": str(75 * idx),
                                                               "y": str(80 * idx),
                                                               "width": "150",
                                                               "height": "70"})
            node_distribution_options[idx] = ElementTree.SubElement(node[idx], "NodeDistributionOptions")
            state_values = df[df.columns[idx]].unique()
            number_states = state_values.size
            # if the variable is categorical
            if number_states <= self.category_threshold:
                variable[idx] = ElementTree.SubElement(node[idx], "Variable", {"id": str(idx),
                                                                               "name": self.nodeNames[idx],
                                                                               "valueType": "discrete"})
                # add the states
                states[idx] = ElementTree.SubElement(variable[idx], "States", {"count": str(number_states)})
                for state_df in state_values:
                    ElementTree.SubElement(states[idx], "State", {"name": str(state_df)})
            # if the variable is continuous
            else:
                variable[idx] = ElementTree.SubElement(node[idx], "Variable", {"id": str(idx),
                                                                               "name": self.nodeNames[idx],
                                                                               "valueType": "continuous"})

                # states[idx] = ElementTree.SubElement(variable[idx], "States", {"count": "2"})
                # ElementTree.SubElement(states[idx], "State", {"name": "True"})
                # ElementTree.SubElement(states[idx], "State", {"name": "False"})

        # ----------- Links -----------------
        links = ElementTree.SubElement(network, "Links", {"count": str(self.Edges)})
        edge_count = 0
        for idx in range(self.V):
            for edge in range(self.graph[idx].__len__()):
                ElementTree.SubElement(links, "Link", {"id": str(edge_count),
                                                       "from": str(idx),
                                                       "to": str(self.graph[idx][edge])})
                edge_count += 1

        tree = ElementTree.ElementTree(network)
        tree.write(network_name, encoding="utf-8", xml_declaration=True)

    def create_structure_from_xml(self, in_file):
        tree = ElementTree.parse(in_file)
        network = tree.getroot()
        names = defaultdict(list)
        for subtree in network:
            if subtree.tag == 'Nodes':
                for node in subtree:
                    names[node.attrib['id']] = node.attrib['name']
            if subtree.tag == 'Links':
                for link in subtree:
                    link_from = link.attrib['from']
                    link_to = link.attrib['to']
                    self.add_edge(list(self.nodeNames).index(names[link_from]),
                                  list(self.nodeNames).index(names[link_to]))

    def has_cycles(self):
        out = False
        self.components.clear()
        self.componentIndex = 0
        # print "SSC in the graph "
        self.scc()
        for idx in range(self.componentIndex):
            if self.components[idx].__len__() >= 2:
                out = True
        return out

    @staticmethod
    def exist(element, list_element):
        try:
            list_element.index(element)
            return True
        except ValueError:
            return False

    def remove_cycle(self):
        # df = self.dataFrame
        while self.has_cycles():
            for idx in range(self.componentIndex):
                if self.components[idx].__len__() >= 2:
                    # calculate the edges in the cycle
                    cycle_edges = defaultdict(list)
                    index = 0
                    for node_y in self.components[idx]:
                        for node_x in self.graph[node_y]:
                            if self.exist(node_x, self.components[idx]):
                                cycle_edges[index].append(node_y)  # position 0
                                cycle_edges[index].append(node_x)  # position 1
                                index += 1
                    # remove from cycle_edges the edge with less dependency
                    # find the edge with less dependency
                    min_dependency = float('inf')
                    min_edge = -1
                    for edge in cycle_edges:
                        ixy = self.dependencies[cycle_edges[edge][1], cycle_edges[edge][0]]

                        if ixy < min_dependency:
                            min_dependency = ixy
                            min_edge = edge
                    # remove edge
                    self.graph[cycle_edges[min_edge][0]].remove(cycle_edges[min_edge][1])

    def create_structure(self, threshold):
        self.Edges = 0
        self.graph = defaultdict(list)  # reset graph for each threshold
        for id_x in range(self.V):
            for id_y in range(id_x + 1, self.V):
                ixy = self.dependencies[id_x, id_y]
                iyx = self.dependencies[id_y, id_x]
                if ixy >= threshold or iyx >= threshold:
                    if ixy >= iyx:
                        self.add_edge(id_y, id_x)
                    else:
                        self.add_edge(id_x, id_y)
        self.remove_cycle()

    def open_dependencies(self, name):
        self.dependencies = np.load("borrar/" + name + "_dependencies.npy")

    def learn_dependencies(self, name, dependency_measure):
        df = self.dataFrame
        for id_x in range(self.V):
            for id_y in range(id_x + 1, self.V):
                x = df.iloc[:, id_x].values
                y = df.iloc[:, id_y].values
                # ------ calculate the optimal number of bins using cd_ixy
                # num_bin_x, num_bin_y = binSelection.calculateBinStatic(x, y, 5)
                # best bins founded for the example
                num_bin_x = 10
                num_bin_y = 10
                # ------ calculate the optimal number of bins using ud_ixy
                # [num_bin_x, num_bin_y] = binSelection.calculateBinStatic(x, y, 4)
                ixy = dependency_measure(x, y, num_bin_x, num_bin_y)
                iyx = dependency_measure(y, x, num_bin_y, num_bin_x)

                self.dependencies[id_x, id_y] = ixy
                self.dependencies[id_y, id_x] = iyx
                # print("bins in " + df.columns[id_x] + ": " + str(num_bin_x) +
                #       " bins in " + df.columns[id_y] + ": " + str(num_bin_y))
                # print("Information in " + df.columns[id_y] + " of " + df.columns[id_x] + " : " + str(ixy))
                # print("Information in " + df.columns[id_x] + " of " + df.columns[id_y] + " : " + str(iyx))

        np.save(name + "_dependencies.npy", self.dependencies)

    def similarity(self, graph_to_compare):
        """

        :type graph_to_compare: Graph
        """
        equal = 0
        in_self = 0
        in_graph_to_compare = graph_to_compare.Edges
        for u in range(self.V):
            for v in self.graph[u]:
                if (v in graph_to_compare.graph[u]) or (u in graph_to_compare.graph[v]):
                    equal += 1
                    # graph_to_compare.graph[u].remove(v)  # this is not need
                    in_graph_to_compare -= 1  # we subtract one element from graph to compare
                else:
                    in_self += 1
        return equal / float(equal+in_graph_to_compare+in_self)


def find_best_threshold(granularity):
    """
    the function return the threshold that generate the best comparative value against the base network
    :type granularity: int
    """
    size = granularity
    comparative_values = np.zeros(size)
    max_value = 0
    best_threshold = 0
    index = 0
    threshold_vector = np.linspace(0, 1, size)
    for threshold_value in threshold_vector:
        proposal_network.create_structure(threshold_value)
        value = base_network.similarity(proposal_network)
        comparative_values[index] = value
        index += 1
        # percentage = (index / float(size))*10
        # if percentage == math.floor(percentage):
        #     print(str(percentage*10))
        if value > max_value:
            max_value = value
            best_threshold = threshold_value
    # plt.plot(threshold_vector, comparative_values, 'b.')
    # plt.show()
    return best_threshold, max_value
# -------------------------------------------------------------------------

# --------- Main function ---------------------------------------

df_data = pd.read_csv(
    "breastCancerComparison-50000samples/50000samples_SyntheticBreastCancerData.csv", na_filter=True)
df_data.dropna(inplace=True)

base_network = Graph(df_data)
proposal_network = Graph(df_data)
base_network.create_structure_from_xml("breastCancerComparison-50000samples/base-network.bayes")

# --------- Comparative of networks already created ----------------------------
# proposal_network.create_structure_from_xml(
#     "breastCancerComparison-150000samples/CD-samples_150000-threshold_0.132413241324-value_0.904761904762.bayes")
# print("Similarity with proposal: " + str(base_network.similarity(proposal_network)))

# -------- Learn or open the network structure from data --------------------
start_time = time.time()
proposal_network.learn_dependencies("test/cd_50000samples_SyntheticBreastCancerData", CorreMeasures.cd_ixy)
# proposal_network.open_dependencies("cd_50000samples_SyntheticBreastCancerData")

threshold_, threshold_value_ = find_best_threshold(10000)
proposal_network.create_structure(threshold_)
print("Execution time: --- %f seconds ---" % (time.time() - start_time))
print("Similarity: " + str(threshold_value_))
proposal_network.to_xml("test/CD-samples_" + str(df_data.shape[0]) +
                        "-threshold_" + str(threshold_) +
                        "-value_" + str(threshold_value_) + ".bayes")

print("End.")
