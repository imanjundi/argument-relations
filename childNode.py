from event import Event
from abc import ABC, abstractmethod

from util import remove_url_and_hashtags


class ChildNode(ABC):
    """
    The abstract class ChildNode. A ChildNode has an ID, a name (which corresponds to a textual representation of the
    content of the node) and a type (e.g. Pro). Each node has information about whether it has children or
    whether it is a leaf node.
    """

    def __init__(self, node_dict):
        """
        Represent the class ChildNode. A ChildNode has the following attributes:
        id: a unique identifier, e.g. 'E-1Q7SF3H-1623' or '1.12'
        name: a name, e.g. 'What would an impartial observer, with access to data from social media, find out about the sources of radicalism and hate speech?'
        type: what kind of node, can be issue, idea, pro, con
        is_leaf: boolean stores whether the node is a leaf
        direct_children: if the node has children; a list of ChildNode objects that are directly attached to this node.
        parent: exactly one node that is the parent of this node (or None if it is the root node)
        siblings: a list that contains the sibling nodes; can be empty of no siblings are available
        embedding: either None if it has not been encoded yet, or an embedding representation of this node
        """
        self.id = node_dict["id"]
        self.name = node_dict["name"]
        self.type = node_dict["type"]
        self.direct_children = self.init_direct_children(node_dict)
        # if child list is empty this node is a leaf node
        self.is_leaf = True if len(self.direct_children) == 0 else False
        self.parent = None
        self.embedding = None
        self.extra_embeddings = {}
        self.siblings = []
        # set this node as parent for each of its' children
        for child in self.direct_children:
            child.parent = self
            # add all other children on the next level to each others sibling list
            for child2 in self.direct_children:
                if child.id != child2.id:
                    child.siblings.append(child2)

    def add_embedding(self, embedding):
        """adds an embedding representation for this node, sets flag to true"""
        self.embedding = embedding

    def lowest_common_subsumer(self, other):
        """
        Extract the lowes common subsumer(s) / lowest common ancestor(s) of the current node and a given one.
        :type other: ChildNode
        :param other: Another ChildNode object the LCS should be computed to.
        :return: the ChildNode that represents the lowes common subsumer or None if it is root
        """
        # self and other are the same? lcs is self
        if self == other:
            return self
        hypernyms_node1 = self.get_all_hypernyms(self, [])
        hypernyms_node2 = self.get_all_hypernyms(other, [])
        # other is a hypernym of self? then other is the lcs
        if other in set(hypernyms_node1):
            return other
        # self is in hypernyms of other? then self is the lcs
        elif self in set(hypernyms_node2):
            return self
        # find the set of common hypernyms and the one with the lowest level is the lowes common subsumer
        common_hypernyms = list(set(hypernyms_node1).intersection(set(hypernyms_node2)))
        # the nodes do not share any common hypernyms, so the lowest subsumer is the root node
        if len(common_hypernyms) <= 1:
            return None
        # the node at the lowest level / the deepest is the *lowest* common subsumer
        levels_hypernyms = [node.get_level() if node else 0 for node in common_hypernyms]
        return common_hypernyms[levels_hypernyms.index(max(levels_hypernyms))]

    def get_all_hypernyms(self, node, hypernyms):
        """recursively iterate trough all nodes of a map and append the parent until root..."""
        if not node:
            return hypernyms
        else:
            parent = node.parent
            hypernyms.append(parent)
            self.get_all_hypernyms(parent, hypernyms)
        return hypernyms

    def get_level(self):
        """Return the level in the tree of the node (the level equals to the number of hypernyms in the graph)"""
        hypernyms = self.get_all_hypernyms(self, [])
        if hypernyms:
            return len(hypernyms)
        else:
            return 0

    def shortest_path(self, other):
        """
        Returns the distance of the shortest path linking the two nodes (if
        one exists). If a node is compared with itself 0 is returned. The distance is denoted by the number of edges
        that exist in the shortest path.
        :param other: a ChildNode to compute the shortest path distance to
        :return: The number of edges in the shortest path connecting the two nodes
        """
        lcs = self.lowest_common_subsumer(other)
        depth_self = self.get_level()
        depth_other = other.get_level()
        # lcs is root
        if lcs == None:
            return depth_self + depth_other
        depth_lcs = lcs.get_level()
        return (depth_self - depth_lcs) + (depth_other - depth_lcs)

    def __str__(self):
        return str(self.name)

    @abstractmethod
    def init_direct_children(self, node_dict) -> list:
        """"""


class DelibChildNode(ChildNode):

    def __init__(self, node_dict):
        """
        DelibChildNode as specific additional attributes that are only present in this kind of Node.
        description: more detailed content of the node.
        events: a list of Events, each associated with its own attributes (see OBJ Event)
        :param node_dict: the json representation of the node
        """
        super(DelibChildNode, self).__init__(node_dict)

        self.description = node_dict["description"]
        self.creator = node_dict["creator"]
        self.events = self.init_events(node_dict["events"])

    def init_direct_children(self, node_dict):
        """Initialize all direct child nodes, returns the empty list if no children"""
        child_list = []
        if "children" in node_dict and node_dict["children"] != None:
            children = node_dict["children"]
            for c in children:
                child_list.append(DelibChildNode(c))
        return child_list

    def init_events(self, event_node):
        """Initialize all events that are connected to this child"""
        event_nodes = []
        for event in event_node:
            event_nodes.append(Event(event))
        return event_nodes

    def number_events(self):
        """Returns the number of events connected to this child"""
        return len(self.events)


class KialoChildNode(ChildNode):
    """
    The Kialo argument map is a simple deliberation map. The nodes can either be of type 'Pro' or type 'Con'
    Each node can have further children. Each node is associated with a tree depth.
    """

    def __init__(self, node_dict):
        """
        The KialoChildNode has one additional attribute
        depth: The tree depth of this node
        :param node_dict: the dictionary representation of the node
        """
        node_dict["name"] = remove_url_and_hashtags(node_dict["name"])
        super(KialoChildNode, self).__init__(node_dict)
        self.created = node_dict["created"]
        self.impact = node_dict["impact"]
        self.edited = node_dict["edited"]
        self.depth = self.get_depth()

    def init_direct_children(self, node_dict):
        """Initialize all direct child nodes, returns the empty list if no children"""
        child_list = []
        if "children" in node_dict and node_dict["children"] != None:
            children = node_dict["children"]
            for c in children:
                child_list.append(KialoChildNode(c))
        return child_list

    def get_depth(self):
        """Returns the depth of this node"""
        return self.id.count(".")
