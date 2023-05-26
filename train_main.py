import argparse
import logging

from sentence_transformers import LoggingHandler

RESULTS_DIR = 'results/'

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def parse_args(add_more_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--debug_size', type=int, default=0)
    parser.add_argument('--debug_maps_size', type=int, default=0)
    parser.add_argument('--do_train', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--do_eval', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--eval_not_trained', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--model_name_or_path', help="model", type=str)
    parser.add_argument('--eval_model_name_or_path', help="model", type=str, default=None)
    parser.add_argument('--output_dir_prefix', type=str, default=None)
    parser.add_argument('--output_dir_label', type=str)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--lang', help="english, italian,..", type=str, default=None)
    parser.add_argument('--max_seq_length', type=int, default=256)
    parser.add_argument('--use_descriptions', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--train_on_one_map',
                        help="either train on `argument_map` and eval on all others or train on all others and evaluate on `argument_map`",
                        type=lambda x: (str(x).lower() == 'true'), default=False)
    if add_more_args:
        add_more_args(parser)
    parser.add_argument('--hard_negatives', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--hard_negatives_size', type=int, default=-1)
    args = vars(parser.parse_args())

    if args['debug_size'] > 0:
        logging.info(f"!!!!!!!!!!!!!!! DEBUGGING with {args['debug_size']} samples")

    if args['debug_maps_size'] > 0:
        logging.info(f"!!!!!!!!!!!!!!! DEBUGGING with {args['debug_maps_size']} maps")

    return args


def get_output_dir(model_name, args, map_label=None):
    model_save_path_prefix = '/'.join([(f'{args["output_dir_prefix"]}' if args['output_dir_prefix'] else ''),
                                       (f"domain{args['training_domain_index']}"
                                        if 'training_domain_index' in args and args['training_domain_index'] >= 0
                                        else '')])
    # + model_name.replace("/", "-")
    model_save_path_prefix = model_save_path_prefix if model_save_path_prefix.startswith(RESULTS_DIR) else (
            RESULTS_DIR + model_save_path_prefix)
    if not map_label:
        return model_save_path_prefix
    return model_save_path_prefix + \
        (f'-{args["output_dir_label"]}' if args['output_dir_label'] else '') + \
        ('-trained' if args['train_on_one_map'] else '-evaluated') + f'-on-{map_label}' + \
        (f'-dev-{args["argument_map_dev"]}' if args['argument_map_dev'] else '')
