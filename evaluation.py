import numpy as np
import torch
from sentence_transformers.util import semantic_search
from torch.fft import Tensor

from childNode import ChildNode


class Evaluation:

    def __init__(self, argument_map, only_parents=False, only_leafs=False, child_node_type=None,
                 candidate_node_types=None, no_ranks=False, close_relatives=False, max_candidates=0):
        """
        Computes the number of times a rank is equal or lower to a given rank.
        :param argument_map [ArgumentMap]: an ArgumentMap object with initialized embedding nodes (embeddings have to normalized!)
        :param only_parents [bool]: if true, only parents of child nodes to be evaluated are considered as candidates
        :param only_leafs [bool]: if true, only child nodes that are leafs, i.e. have no children are considered for evaluation
        :param child_node_type [str]: if specified, only child nodes of a given type are considered for evaluation
        :param candidate_node_types [set]: if given, has to be given as a set: only nodes with the speicified node types
        are considered as candidates
        """
        # argument map with encoded nodes
        self.argument_map = argument_map
        # gather all nodes in the map and construct a node2index and an embedding matrix
        self.all_nodes = self.argument_map.all_nodes
        # extract the nodes to be tested
        self.child_nodes: list[ChildNode] = self.get_child_nodes(only_leafs, child_node_type, candidate_node_types)
        # extract their corresponding parents
        self.parent_nodes = [child.parent for child in self.child_nodes]
        # extract possible candidate (all parents must be within the candidates)
        self.candidate_nodes: list[ChildNode] = self.get_candidate_nodes(only_parents, candidate_node_types)
        # consider close relatives when computing the metrics
        self.close_relatives = close_relatives
        self.max_candidates = max_candidates
        assert not close_relatives, "close_relatives is not supported"
        assert len(self.child_nodes) == len(
            self.parent_nodes), "the number of children and their parents is not the same"
        self.ranks, self.predictions = (None, None) if no_ranks else self.compute_ranks()

    def compute_ranks(self):
        if len(self.child_nodes) == 0:
            return [], []
        nodes_embedding = torch.from_numpy(np.array([x.embedding for x in self.child_nodes]))
        candidates_embedding = torch.from_numpy(np.array(
            [x.extra_embeddings['parent'] for x in self.candidate_nodes] if self.candidate_nodes[0].extra_embeddings
            else [x.embedding for x in self.candidate_nodes]))
        candidates = [{'id': x.id} for x in self.candidate_nodes]
        return Evaluation.eval_nodes(nodes_embedding, candidates_embedding, candidates,
                                     [x.id for x in self.child_nodes], [x.parent.id for x in self.child_nodes],
                                     top_k=10)

    def get_child_nodes(self, only_leafs, child_node_type, candidate_node_types):
        """Extract the child nodes to be used for evaluation. Apply filtering rules if specified."""
        # case 1: I want to test all possible child nodes in this map
        child_nodes = self.argument_map.child_nodes
        # case 2: I only want to test leaf nodes (= the nodes that were added 'the latest')
        if only_leafs:
            child_nodes = [node for node in child_nodes if node.is_leaf]
        # case 3: I want to test only specific node types
        if child_node_type:
            child_nodes = [node for node in child_nodes if node.type == child_node_type]
        # case 4: I want to test only nodes with certain parent node types
        if candidate_node_types:
            child_nodes = [node for node in child_nodes if node.parent.type in candidate_node_types]
        return child_nodes

    def get_candidate_nodes(self, only_parents, candidate_node_types):
        """Extract the candidate nodes to be used for evaluation. Apply filtering rules if specified."""
        # case 1: consider all nodes of a map as candidates
        candidate_nodes = self.all_nodes
        # case 2: consider only parents as candidates
        if only_parents:
            candidate_nodes = self.parent_nodes
        # filter out candidates of certain types if that is specified
        if candidate_node_types:
            candidate_nodes = [node for node in candidate_nodes if node.type in candidate_node_types]
        return candidate_nodes

    @staticmethod
    def eval_nodes(node_embedding: Tensor, candidates_embedding: Tensor, candidates: list[dict],
                   node_ids: list[str], parent_ids: list[str], top_k=0):
        if top_k == 0:
            top_k = len(candidates)
        hits_list = semantic_search(node_embedding, candidates_embedding,
                                    top_k=len(candidates))
        all_predictions, all_ranks = [], []
        for hits, node_id, parent_id in zip(hits_list, node_ids, parent_ids):
            predictions = []
            rank = -1
            i = 1
            for hit in hits:
                candidate = candidates[hit['corpus_id']].copy()
                if candidate['id'] == node_id:
                    continue
                candidate['score'] = hit['score']
                if candidate['id'] == parent_id:
                    rank = i
                if len(predictions) < top_k:
                    predictions.append(candidate)
                elif rank != -1:
                    break
                i += 1
            all_ranks.append(rank)
            all_predictions.append(predictions)
        return all_ranks, all_predictions

    @staticmethod
    def calculate_metrics(ranks):
        return {'mrr': Evaluation.mean_reciprocal_rank(ranks),
                'p5': Evaluation.precision_at_rank(ranks, 5),
                'p1': Evaluation.precision_at_rank(ranks, 1)}

    @staticmethod
    def precision_at_rank(ranks, k):
        """
        Computes the number of times a rank is equal or lower to a given rank.
        :param k: the rank for which the precision is computed
        :return: the precision at a certain rank (float)
        """
        assert k >= 1
        correct = len([rank for rank in ranks if rank <= k])
        return correct / len(ranks)

    @staticmethod
    def mean_reciprocal_rank(ranks):
        """
        Computes the mean reciprocal rank for a list of ranks. (As we only have one relevant item, this equals to MAP)
        :param ranks: a list of ranks
        :return: the mean reciprocal rank
        """
        precision_scores = sum([1 / r for r in ranks])
        return precision_scores / len(ranks)

    def average_taxonomic_distance(self, quartile):
        """
        Computes the average taxonomic distance of the predicted parent nodes for a given quartiles
        :param predictions: a list of predicted parent nodes. the average taxonomic distance to the predicted "parents"
        corresponds to the second quartile
        :return: the quartiles
        """
        self.taxonomic_distances = []
        for i in range(len(self.child_nodes)):
            self.taxonomic_distances.append(self.child_nodes[i].shortest_path(
                self.argument_map.all_nodes_dict[self.predictions[i][0]['id']]))
        return np.quantile(self.taxonomic_distances, quartile, interpolation='midpoint')
