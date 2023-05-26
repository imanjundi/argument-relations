import json
import re
from abc import ABC, abstractmethod
import pickle as pkl
import networkx as nx
from childNode import DelibChildNode, KialoChildNode, ChildNode


class ArgumentMap(ABC):
    """
    Abstract class for Argument Maps. Specific ArgumentMaps can inherit from this base class. The base class
    defines the core of every argument map, which is a method to load the data from source, children, a name and an ID.
    An ArgumentMap is a directed Graph with node objects. Ever node has exactly one parent (except for the root node) and
    can have zero to several children.
    """

    def __init__(self, data_path, label=None):
        """
        Initialize the Map given the path to the data to load from. Every map has the following attributes:
        id: a unique identifier, e.g. 'E-1R8DN19-3171'
        name: a name, e.g. 'Scholio DC2: New Zealand Massacre Highlights Global Reach of White Extremism'
        Args:
            data_path: path to the file that contains the data of the argument map.
        """
        self.label = label
        self.data = self.load_data(data_path)
        self.id = self.data["id"]
        self.name = self.data["name"].strip()
        self.direct_children = self.init_children()
        all_nodes = []
        # create a list that stores all nodes of a map by iterating through the first level of nodes
        # and calling the recursive method for each child.
        for child in self.direct_children:
            all_nodes = self.get_all_children(node=child, child_list=all_nodes)
        self.all_nodes = all_nodes
        self.all_nodes_dict = {x.id: x for x in all_nodes}
        self.child_nodes: list[ChildNode] = [node for node in self.all_nodes if node.parent]
        self.parent_nodes: list[ChildNode] = [child.parent for child in self.child_nodes]

    @abstractmethod
    def load_data(self, data_path) -> dict:
        """This method should read data from a file and return a dictionary object storing all details needed
        to construct a map."""

    @abstractmethod
    def init_children(self) -> list:
        """Method to initialize the Nodes that are directly attached to the root. Should return a list of ChildNodes"""

    def get_all_children(self, node, child_list):
        """recursively iterate trough all nodes of a map and append the children and their children..."""
        child_list.append(node)
        if node.is_leaf:
            return child_list
        else:
            for childnode in node.direct_children:
                self.get_all_children(childnode, child_list)
        return child_list

    def number_of_children(self):
        """Returns the number of child nodes in the map"""
        return len(self.all_nodes)

    def __str__(self):
        return str(self.name)


class DeliberatoriumMap(ArgumentMap):
    """
    An ArgumentMap for Maps from Deliberatorium
    """

    def __init__(self, data_path, label=None):
        """
        Additional attributes for this type of Map are:
        description: a more detailed description, can be 'None'
        :param data_path: the path to the json file of the argument map
        """
        super(DeliberatoriumMap, self).__init__(data_path, label)
        self.description = self.data["description"]

    def load_data(self, json_file):
        """Loads the json object from the json file"""
        try:
            with open(json_file, encoding='utf-8') as f:
                json_obj = json.load(f)
        except json.decoder.JSONDecodeError:
            with open(json_file, encoding='utf-8-sig') as f:
                json_obj = json.load(f)
        return json_obj

    def init_children(self):
        """Initializes the first level of children = all nodes that are directly located at the root of the map"""
        children_list = []
        if self.data["children"]:
            for child in self.data["children"]:
                children_list.append(DelibChildNode(child))
        return children_list


class KialoMap(ArgumentMap):

    def __init__(self, data_path, label=None):
        super(KialoMap, self).__init__(data_path, label)
        self.id = int(self.id)
        self.max_depth = self.get_max_depth()

    def load_data(self, data_path):
        """Loads the data from the .txt files"""
        data_dict = {}
        with open(data_path, 'rb') as f:
            data = pkl.load(f)
        id = data_path.split("/")[-1].replace(".pkl", "")
        claim = data.node["%s.0" % id]["text"].strip()
        self.root_id = "%s.0" % id
        edges = data.edge
        if claim == "":
            if "%s.1" % id not in data.node:
                for k, v in data.node.items():
                    if v["relation"] == 0 and k != "%s.0" % id:
                        claim = v["text"].strip()
                        self.root_id = k
            else:
                claim = data.node["%s.1" % id]["text"].strip()
                self.root_id = "%s.1" % id

        # the name is the topic of the map
        data_dict["name"] = claim
        data_dict["id"] = id
        data_dict["children"] = []
        for nodeid, nodecontent in data.node.items():
            data_dict["children"].append({"id": nodeid, "type": nodecontent["relation"], "name": nodecontent["text"],
                                          "edited": nodecontent["edited"], "impact": nodecontent["votes"],
                                          "created": nodecontent["created"]})

        # for each child the list of direct children has to be added to the node dictionary (which can only
        # be retrieved after having read the file completely.
        for child in data_dict["children"]:
            child["children"] = self.get_direct_children(child["id"], data_dict["children"], edges)
        return data_dict

    def init_children(self):
        """Initializes the first level of children = all nodes that are directly located at the root of the map"""
        children_list = []
        if self.data["children"]:
            for child in self.data["children"]:
                if child["id"] == self.root_id:
                    self.root = KialoChildNode(child)
                    for rootchild in child["children"]:
                        children_list.append(KialoChildNode(rootchild))
        return children_list

    def get_direct_children(self, id, all_nodes, edges):
        """Given an id, extract all child nodes that correspond to direct children of a node"""
        child_ids = [k for k, v in edges.items() if id in v]
        direct_children = [node for node in all_nodes if
                           node["id"] in child_ids]
        return direct_children

    def get_max_depth(self):
        """Return the maximal tree depth of this map"""
        return max([node.get_level() for node in self.all_nodes])

    @staticmethod
    def remove_links(sentence):
        """Removes all links and empty brackets '()' from a text."""
        pattern = "((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"
        if re.search(pattern, sentence):
            sentence = sentence.replace(re.search(pattern, sentence).group(0), "")
        if re.search("\(\)", sentence):
            sentence = sentence.replace(re.search("\(\)", sentence).group(0), "")
        sentence = sentence.replace("[", "").replace("]", "")
        return sentence
