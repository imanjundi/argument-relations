import logging

from evaluation import Evaluation
from rerank_evaluation import RerankEvaluation

METRICS = ['mrr', 'p5', 'p1', 'dist']


def evaluate_map(encoder, argument_map, node_types, max_candidates=0, cross_encoder=None,
                 eval_types=None):
    if eval_types is None:
        eval_types = ['only_leafs_limited_types']
    results = {}
    node_results = {}
    print(f'eval {argument_map.id} {len(argument_map.all_nodes)=} \n{argument_map.name}')
    encoder.encode_argument_map(argument_map)

    eval_args = {'argument_map': argument_map, 'only_leafs': False, 'max_candidates': max_candidates}
    if cross_encoder:
        eval_args['cross_encoder'] = cross_encoder
    if 'all' in eval_types:
        print("default setting: all nodes are evaluated, all nodes are considered as candidates")
        results['all'], node_results['all'] = eval_one(eval_args)

    eval_args['only_leafs'] = True
    if 'only_leafs' in eval_types:
        print("only check for leaf nodes")
        results['only_leafs'], node_results['only_leafs'] = eval_one(eval_args)

    eval_args['candidate_node_types'] = node_types
    if 'only_leafs_limited_types' in eval_types:
        print("only leaf nodes and only issues and ideas as parents")
        results['only_leafs_limited_types'], node_results['only_leafs_limited_types'] = eval_one(eval_args)

    return results, node_results


def eval_one(eval_args: dict):
    evaluation = create_evaluation(eval_args)
    if len(evaluation.child_nodes) == 0:
        logging.warning('no child nodes found')
        return None, None
    metrics = Evaluation.calculate_metrics(evaluation.ranks)
    metrics['dist'] = evaluation.average_taxonomic_distance(0.5)
    # print(eval.ranks)
    print(f"child nodes: {len(evaluation.child_nodes)} candidates :{len(evaluation.candidate_nodes)}. " +
          format_metrics(metrics))
    node_results = [{'id': c.id, 'parent_id': c.parent.id, 'rank': r, 'dist': t, 'predictions': p} for c, r, p, t in
                    zip(evaluation.child_nodes, evaluation.ranks, evaluation.predictions, evaluation.taxonomic_distances
                        )]
    return metrics, node_results



def create_evaluation(eval_args: dict):
    evaluation_class = RerankEvaluation if 'cross_encoder' in eval_args else Evaluation
    return evaluation_class(**eval_args)


def format_metrics(metrics):
    return ' , '.join([f"{k}: {v:.2f}" for k, v in metrics.items()])
