import torch
from sentence_transformers import util

from childNode import ChildNode
from evaluation import Evaluation


class RerankEvaluation(Evaluation):
    def __init__(self, argument_map, cross_encoder, only_parents=False, only_leafs=False, child_node_type=None,
                 candidate_node_types=None, no_ranks=False, close_relatives=False, max_candidates=0):
        self.cross_encoder = cross_encoder
        super().__init__(argument_map, only_parents, only_leafs, child_node_type, candidate_node_types, no_ranks,
                         close_relatives, max_candidates)

    def compute_ranks(self):
        if not self.child_nodes:
            return None, None
        top_k = 32
        all_hits = util.semantic_search(get_embeddings(self.child_nodes), get_embeddings(self.candidate_nodes),
                                        top_k=top_k)
        ranks, predictions = [], []
        for child_node_idx, hits in enumerate(all_hits):
            cross_inp = [[self.child_nodes[child_node_idx].name,
                          self.candidate_nodes[hit['corpus_id']].name] for hit in hits]
            cross_scores = self.cross_encoder.predict(cross_inp, show_progress_bar=False)
            for idx in range(len(cross_scores)):
                hits[idx]['cross-score'] = cross_scores[idx]
            # hits = sorted(hits, key=lambda x: x['score'], reverse=True)
            hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
            pred = [self.candidate_nodes[hit['corpus_id']] for hit in hits]
            predictions.append(self.candidate_nodes[hits[0]['corpus_id']])
            hits = [hit for hit in hits
                    if self.child_nodes[child_node_idx].id != self.candidate_nodes[hit['corpus_id']].id]
            ranks.append(next((i for i, hit in enumerate(hits) if
                               self.candidate_nodes[hit['corpus_id']].id == self.parent_nodes[child_node_idx].id),
                              top_k) + 1)
        return ranks, predictions


def get_embeddings(nodes: list[ChildNode]):
    return torch.Tensor([x.embedding for x in nodes])
