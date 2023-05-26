import copy
import faulthandler
import itertools
import json
import logging
import math
import os
import pickle
import random
import re
import shutil
import signal
import statistics
from pathlib import Path

import wandb
from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample, CrossEncoder
from sentence_transformers import models, losses, datasets
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed

import templates
import util
from argumentMap import KialoMap
from childNode import ChildNode
from data_loader import SameMapPerBatchDataLoader, validate_for_no_duplicates_batch
from eval_util import METRICS, evaluate_map, format_metrics
from encode_nodes import MapEncoder
from evaluation import Evaluation
from kialo_domains_util import get_maps2uniquetopic
from kialo_util import read_data, read_annotated_maps_ids, read_annotated_samples, save_maps
from templates import format_primary
from util import remove_url_and_hashtags, sample
from train_main import parse_args, get_output_dir, RESULTS_DIR

AVAILABLE_MAPS = ['dopariam1', 'dopariam2', 'biofuels', 'RCOM', 'CI4CG']

logging.basicConfig(format='%(asctime)s,%(msecs)d p%(process)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def add_more_args(parser):
    parser.add_argument('--debug_map_index', type=str, default=None)
    parser.add_argument('--no_data_split', type=str, default=None)
    parser.add_argument('--training_domain_index', type=int, default=-1)
    parser.add_argument('--train_method', choices=['mulneg', 'class', 'cossco'], default='mulneg')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--train_negatives_size', type=int, default=20)
    parser.add_argument('--train_maps_size', type=int, default=0)
    parser.add_argument('--train_per_map_size', type=int, default=0)
    parser.add_argument('--batch_from_same_map', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--strict_batch_size', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--data_samples_seed', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_templates', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--template_id', type=str, default='beginning')
    parser.add_argument('--template_not', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--annotated_samples_in_test', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--use_dev', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--max_candidates', type=int, default=0)
    parser.add_argument('--do_eval_annotated_samples', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--save_embeddings', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--save_detailed_results', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--rerank', type=lambda x: (str(x).lower() == 'true'), default=False)


def main():
    faulthandler.register(signal.SIGUSR1.value)

    args = parse_args(add_more_args)

    seed = args['seed']
    set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    model_name = args['model_name_or_path']
    max_seq_length = args['max_seq_length']
    num_epochs = args['num_train_epochs']

    output_dir = get_output_dir(model_name, args)
    model_save_path = output_dir + '/' + 'model'
    logging.info(f'{output_dir=}')
    logging.getLogger().handlers[0].flush()

    if args['local']:
        os.environ['WANDB_MODE'] = 'disabled'
    experiment_name = output_dir.removeprefix(RESULTS_DIR)
    wandb.init(project='argument-maps',
               name=experiment_name,
               group=re.sub(r'-seed\d+', '', experiment_name) if '-seed' in experiment_name else experiment_name,
               # to fix "Error communicating with wandb process"
               # see https://docs.wandb.ai/guides/track/launch#init-start-error
               settings=wandb.Settings(start_method="fork"))
    wandb.config.update(args | {'data': 'kialoV2', 'train_negative_class_size': args['train_negatives_size']})

    util.args = args
    # save args
    path = Path(output_dir) / 'data'
    path.mkdir(exist_ok=True, parents=True)
    (path / f'args.json').write_text(json.dumps(args))

    templates.init()

    data_splits = None
    main_domains = []
    if args['do_train'] or args['do_eval']:
        argument_maps = read_data(args)

        if args['training_domain_index'] >= 0:
            maps2uniquetopic, (_, _, main2subtopic) = get_maps2uniquetopic('data/kialoID2MainTopic.csv',
                                                                           'data/kialo_domains.tsv')
            main_domains = list(main2subtopic.keys())

            # domain_argument_maps = {domain: [KialoMap(str(data_path / (map_name + '.txt')), map_name)
            #                                  for map_name, map_domain in maps2uniquetopic.items() if map_domain == domain]
            #                         for domain in main2subtopic}
            domain_argument_maps = {domain: [] for domain in main2subtopic}
            for argument_map in argument_maps:
                if argument_map.id in maps2uniquetopic:
                    domain_argument_maps[maps2uniquetopic[argument_map.id]].append(argument_map)
                else:
                    logging.warning(f'{argument_map.label} {argument_map.name} skipped!')
            argument_maps = domain_argument_maps[main_domains[args['training_domain_index']]]
            args['training_domain'] = main_domains[args['training_domain_index']]
            logging.info(f"{args['training_domain']=}")
            logging.info(f"{len(argument_maps)=} maps in domain args['training_domain_index']={args['training_domain']}")
            wandb.config.update(args | {'data': 'kialoV2'})

        data_splits = split_data(argument_maps, args, output_dir, seed)

    if args['do_train']:
        if args['train_maps_size']:
            logging.info(f"{args['train_maps_size']=}")
            data_splits['train'] = util.sample(
                # filter to maps with #children strictly more than needed (root can't be used)
                [x for x in data_splits['train'] if len(x.child_nodes) > args['train_per_map_size']],
                args['train_maps_size'])

        maps_samples = prepare_samples(data_splits['train'], 'train', args, output_dir)
        maps_samples_dev = prepare_samples(data_splits['dev'], 'dev', args, output_dir)

        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        train_dataloader, train_loss = prepare_training(maps_samples, model, args)

        (path / f'args.json').write_text(json.dumps(args | {'actual_train_batch_size': train_dataloader.batch_size}))

        dev_samples = list(itertools.chain(*maps_samples_dev.values()))
        dev_evaluator = (EmbeddingSimilarityEvaluator.from_input_examples(
            dev_samples, batch_size=args['train_batch_size'], name='dev')
                         if args['use_dev'] else None)

        assert len(train_dataloader)
        logging.info(f'{len(train_dataloader)=}')
        logging.getLogger().handlers[0].flush()
        # 10% of train data for warm-up
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
        logging.info("Warmup-steps: {}".format(warmup_steps))

        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  epochs=num_epochs,
                  evaluator=dev_evaluator,
                  evaluation_steps=args['eval_steps'] if args['eval_steps'] else
                  (int(len(train_dataloader) * 0.1) if dev_evaluator else 0),
                  warmup_steps=warmup_steps,
                  optimizer_params={'lr': args['lr']},
                  output_path=model_save_path,
                  use_amp=False  # Set to True, if your GPU supports FP16 operations
                  )

    # eval
    if args['do_eval'] or args['do_eval_annotated_samples'] or args['save_embeddings']:
        model = SentenceTransformer(args['eval_model_name_or_path'] if args['eval_model_name_or_path'] else
                                    model_save_path)
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') if args['rerank'] else None
        if args['do_eval']:
            map_encoder = MapEncoder(max_seq_len=args['max_seq_length'],
                                     sbert_model_identifier=None,
                                     model=model,
                                     normalize_embeddings=True,
                                     use_templates=args['use_templates'])
            all_results = []
            all_results.extend(
                eval(output_dir, data_splits['test'],
                     domain=main_domains[args['training_domain_index']] if args['training_domain_index'] >= 0 else 'all',
                     map_encoder=map_encoder, cross_encoder=cross_encoder, args=args))
            if args['training_domain_index'] >= 0:
                for domain in main_domains[:args['training_domain_index']] + main_domains[args['training_domain_index']+1:]:
                    all_results.extend(eval(output_dir, domain_argument_maps[domain], domain=domain,
                                            map_encoder=map_encoder, cross_encoder=cross_encoder, args=args))
            avg_results = get_avg(all_results)
            (Path(output_dir) / f'results/avg.json').write_text(json.dumps(avg_results))
            wandb.log({'test': {'avg': avg_results}})

        if args['do_eval_annotated_samples']:
            eval_samples(output_dir, args, model, cross_encoder)

    if args['do_train'] and (args['train_maps_size'] or args['train_per_map_size']):
        # remove saved model
        shutil.rmtree(Path(model_save_path))


def split_data(argument_maps: list[KialoMap], args: dict, output_dir: str, seed: int):
    test_size = 0.2
    data_splits = {}
    annotated_maps, remaining_maps = [], argument_maps
    if args['annotated_samples_in_test']:
        annotated_maps_ids = read_annotated_maps_ids(args['local'])
        not_annotated_maps, annotated_maps = [], []
        for argument_map in argument_maps:
            (not_annotated_maps, annotated_maps)[argument_map.id in annotated_maps_ids].append(argument_map)

        more_test_maps_num = round(test_size * len(argument_maps) - len(annotated_maps))
        logging.info(f'keep annotated samples in test: {len(annotated_maps)=} + {more_test_maps_num} for {test_size=}')
        # change test_size from percentage to absolute number of maps to include beside the annotated samples
        test_size = more_test_maps_num
        remaining_maps = not_annotated_maps

    data_splits['train'], data_splits['test'] = (
        train_test_split(remaining_maps, test_size=test_size, random_state=seed)
        if not args['no_data_split'] else (remaining_maps, remaining_maps))
    data_splits['test'] = annotated_maps + data_splits['test']

    if args['use_dev']:
        data_splits['train'], data_splits['dev'] = train_test_split(data_splits['train'],
                                                                    test_size=0.2, random_state=seed)
    else:
        data_splits['dev'] = []
    logging.info('train/dev/test using sizes: ' + ' '.join([f'{k}={len(v)} ({(len(v) / len(argument_maps)):.2f})'
                                                            for k, v in data_splits.items()]))

    # save split ids
    path = Path(output_dir) / 'data'
    path.mkdir(exist_ok=True, parents=True)
    for split_name, split in data_splits.items():
        (path/f'{split_name}.json').write_text(json.dumps([x.id for x in split]))

    return data_splits


def prepare_samples(argument_maps, split_name, args, output_dir):
    prepare_func = prepare_training_samples if split_name == 'train' else prepare_dev_samples
    maps_samples = {x.label: [] for x in argument_maps}
    get_samples_from_map = ((lambda x: util.sample(x, args['train_per_map_size']))
                            if args['train_per_map_size'] else (lambda x: x))
    for i, argument_map in enumerate(tqdm(argument_maps, f'preparing samples {split_name}')):
        argument_map_util = Evaluation(argument_map, no_ranks=True, max_candidates=args['max_candidates'])
        all_parents = list(dict.fromkeys(x for x in argument_map_util.parent_nodes))
        for child, parent in get_samples_from_map(list(
                zip(argument_map_util.child_nodes, argument_map_util.parent_nodes))):
            non_parents = [x for x in all_parents if x != parent]
            maps_samples[argument_map.label].extend(prepare_func(child, parent, non_parents, args))

    if args['debug_size']:
        maps_samples = {k: x[:args['debug_size']] for k, x in maps_samples.items()}

    path = Path(output_dir) / 'data'
    path.mkdir(exist_ok=True, parents=True)
    (path / f'{split_name}-samples.json').write_text(json.dumps({k: [vars(x) for x in v[:100]]
                                                                 for k, v in list(maps_samples.items())[:100]}))
    return maps_samples


def prepare_training_samples(child, parent, non_parents, args):
    if args['train_method'] == 'mulneg':
        if not args['hard_negatives']:
            return create_all_possible_examples(child, parent, args['use_templates'], label=1) + \
                   (create_all_possible_examples(child, sample(non_parents, 1)[0], args['use_templates'], label=1)
                    if args['template_not'] else [])
        else:
            if args['use_templates']:
                raise NotImplementedError('hard negatives not supported')
            if len(non_parents) > args['hard_negatives_size'] > 0:
                non_parents = random.sample(non_parents, args['hard_negatives_size'])
                return [create_primary_example([child, parent, non_parent], use_templates=args['use_templates'])
                        for non_parent in non_parents]
    elif args['train_method'] in ['class', 'cossco']:
        return create_all_possible_examples(child, parent, label=1 if args['train_method'] == 'class' else 1.0,
                                            use_templates=args['use_templates']) + \
               list(itertools.chain.from_iterable(
                   create_all_possible_examples(child, non_parent,
                                                label=0 if args['train_method'] == 'class' else 0.0,
                                                use_templates=args['use_templates'])
                   # add negatives (move later to dataloader?)
                   for non_parent in sample(non_parents, args['train_negatives_size'])))


def prepare_dev_samples(child, parent, non_parents, args):
    return [create_primary_example([child, non_parent], label=0 if args['train_method'] == 'class' else 0.0,
                                   use_templates=args['use_templates']) for non_parent in non_parents] + \
           [create_primary_example([child, parent], label=1 if args['train_method'] == 'class' else 1.0,
                                   use_templates=args['use_templates'])]


def create_all_possible_examples(child_node: ChildNode, parent_node: ChildNode, use_templates, label):
    node_types = {1: 'pro', -1: 'con'}
    templated_child, templated_parent = [
        templates.format_all_possible(x.name, parent_node.name, t, use_templates, label) for x, t in
        zip([child_node, parent_node], [node_types[child_node.type], 'parent'])]
    return [InputExample(texts=[c, p], label=label) for c, p in itertools.product(templated_child, templated_parent)]


def create_primary_example(nodes: list[ChildNode], use_templates, label=0):
    types = ['child'] + ['parent'] * (len(nodes) - 1)
    return InputExample(texts=[templates.format_primary(x.name, t, use_templates) for x, t in zip(nodes, types)],
                        label=label)


def prepare_training(maps_samples, model, args):
    train_samples = list(itertools.chain(*maps_samples.values()))
    if args['train_method'] == 'mulneg':
        possible_batch_size = validate_for_no_duplicates_batch(train_samples, args['train_batch_size'],
                                                               args['strict_batch_size'])
        batch_size = min(possible_batch_size, args['train_batch_size'])
        logging.info(f"{possible_batch_size=} vs {args['train_batch_size']=} => actual {batch_size=}")
        train_dataloader = (SameMapPerBatchDataLoader(list(maps_samples.values()), batch_size=batch_size)
                            if args['batch_from_same_map'] else
                            datasets.NoDuplicatesDataLoader(train_samples, batch_size=batch_size))
    else:
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args['train_batch_size'])
    if args['train_method'] == 'mulneg':
        train_loss = losses.MultipleNegativesRankingLoss(model)
    elif args['train_method'] == 'class':
        train_loss = losses.SoftmaxLoss(model=model,
                                        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                        num_labels=2)
    elif args['train_method'] == 'cossco':
        train_loss = losses.CosineSimilarityLoss(model)
    else:
        raise NotImplementedError(f"{args['train_method']}")
    logging.info("Train samples: {}".format(len(train_samples)))
    return train_dataloader, train_loss


def eval(output_dir, argument_maps: list[KialoMap], domain, map_encoder: MapEncoder, cross_encoder: CrossEncoder, args):
    results_path = Path(output_dir) / 'results' / domain
    results_path.mkdir(exist_ok=True, parents=True)
    all_results = []
    maps_all_results = {}
    nodes_all_results = {}
    try:
        for j, argument_map in enumerate(tqdm(argument_maps, f'eval maps in domain {domain}')):
            try:
                results, nodes_all_results[argument_map.label] = evaluate_map(map_encoder,
                                                                              argument_map, {1, -1},
                                                                              max_candidates=args['max_candidates'],
                                                                              cross_encoder=cross_encoder)
            except Exception as e:
                logging.error('cannot evaluate map ' + argument_map.label)
                raise e
            maps_all_results[argument_map.label] = results
            all_results.append(results)
    finally:
        if args['save_detailed_results']:
            (results_path / f'all_maps.json').write_text(json.dumps(maps_all_results))
            (results_path / f'all_nodes.json').write_text(json.dumps(nodes_all_results))
        # wandb.log({'test': maps_all_results})
        # wandb.log({'test': all_results})
        # data = [[map_name.rsplit('-', 1)[-1], v] for map_name, v in maps_all_results.items()]
        # table = wandb.Table(data=data, columns=["map id", "scores"])
        # wandb.log({'test': {'detailed': wandb.plot.line(
        #     table, "map id", "score", title="Detailed results per map id")}})

        if args['save_embeddings']:
            logging.info('saving annotated maps with embeddings')
            annotated_maps_ids = read_annotated_maps_ids(args['local'])
            save_maps([x for x in argument_maps if x.id in annotated_maps_ids], (results_path / f'all_maps_embedding.pkl'))

    avg_results = get_avg(all_results)
    (results_path / f'avg.json').write_text(json.dumps(avg_results))
    wandb.log({'test': {domain: {'avg': avg_results}}})
    return all_results


def eval_samples(output_dir, args, encoder: SentenceTransformer, cross_encoder: CrossEncoder):
    results_path = Path(output_dir) / 'results'
    results_path.mkdir(exist_ok=True, parents=True)

    samples = read_annotated_samples(args['local'], args)
    embeddings = copy.deepcopy(samples) if args['save_embeddings'] else None
    for node_id, sample in tqdm(samples.items(), desc='encode and evaluate annotated samples'):
        sample['text'] = remove_url_and_hashtags(sample['text'])
        candidates = []
        for candidate_id, candidate in sample['candidates'].items():
            candidates.append({'text': remove_url_and_hashtags(candidate['text']), 'id': candidate_id})

        node_embedding = encoder.encode(format_primary(sample['text'], 'child', args['use_templates']), convert_to_tensor=True,
                                        show_progress_bar=False)
        candidates_embedding = encoder.encode([format_primary(x['text'], 'parent', args['use_templates']) for x in candidates],
                                              convert_to_tensor=True,
                                              show_progress_bar=False)

        ranks, predictions = Evaluation.eval_nodes(node_embedding, candidates_embedding, candidates,
                                                   [node_id], [sample['parent ID']])
        sample['rank'], sample['predictions'] = ranks[0], predictions[0]

        if args['save_embeddings']:
            embeddings[node_id]['embedding'] = node_embedding.cpu().detach().numpy()
            embeddings[node_id]['candidates_embedding'] = candidates_embedding.cpu().detach().numpy()

    metrics = Evaluation.calculate_metrics([x['rank'] for x in samples.values()])
    logging.info(format_metrics(metrics))
    (results_path / 'annotated_samples_predictions.json').write_text(json.dumps(samples))
    (results_path / 'annotated_samples_metrics.json').write_text(json.dumps(metrics))
    wandb.log({'samples': metrics})

    if args['save_embeddings']:
        with open((results_path / 'annotated_samples_embeddings.pkl'), 'wb') as f:
            pickle.dump(embeddings, f)


def get_avg(all_results):
    avg_results = {
        key: {inner_key: statistics.fmean(entry[key][inner_key] for entry in all_results if entry[key])
              for inner_key in METRICS}
        for key, value in all_results[0].items()}
    return avg_results


if __name__ == '__main__':
    main()
