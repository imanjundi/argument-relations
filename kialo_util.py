import json
import os
import pickle
import sys

import pandas as pd

from pathlib import Path
from argumentMap import KialoMap
from tqdm.autonotebook import tqdm

LANGS = ['english', 'french', 'german', 'italian', 'other']


def read_data(args):
    processed_maps_path = Path('temp/maps.pkl')

    if maps := read_maps(processed_maps_path):
        return maps

    data_path = get_base_data_path(args['local']) / 'kialoV2'

    assert args['lang'] in [*LANGS, None]

    # list of maps with no duplicates
    maps = []
    for lang in ([args['lang']] if args['lang'] else LANGS):
        print(f'{lang=}')
        maps += [x for x in data_path.glob(f'{lang}/*.pkl') if x.stem not in [y.stem for y in maps]]

    if args['debug_maps_size']:
        maps = sorted(maps, key=os.path.getsize)
        if args['debug_map_index']:
            maps = list(data_path.glob(f"**/{args['debug_map_index']}.pkl")) + \
                   [x for x in maps if x.stem != args['debug_map_index']]
        maps = maps[:args['debug_maps_size']]

    argument_maps = [KialoMap(str(_map), _map.stem) for _map in tqdm(maps, f'processing maps')
                     # some maps seem to be duplicates with (1) in name
                     if '(1)' not in _map.stem]
    print(f'remaining {len(maps)} maps after clean up')

    save_maps(argument_maps, processed_maps_path)

    return argument_maps


def read_maps(path):
    if path.exists():
        # fix for pickle
        # RecursionError: maximum recursion depth exceeded while calling a Python object
        recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)
        print(f'reading processed maps from {path}')
        with (open(path, 'rb')) as f:
            maps = pickle.load(f)
        sys.setrecursionlimit(recursion_limit)
        return maps
    return None


def save_maps(argument_maps, path):
    # fix for pickle
    # RecursionError: maximum recursion depth exceeded while calling a Python object
    recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)
    path.parent.mkdir(exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(argument_maps, f)
    sys.setrecursionlimit(recursion_limit)


def read_annotated_maps_ids(local: bool):
    data_path = get_annotation_data_path(local)
    annotated_maps_df = pd.read_csv(data_path / 'all_maps.csv', sep='\t')
    ids = annotated_maps_df['mapID'].to_list()
    return ids


def read_annotated_samples(local: bool, args: dict = None):
    data_path = get_annotation_data_path(local)
    data = json.loads((data_path / 'child_and_candidate_info.json').read_text())
    # clean up instances where the child node is in candidates
    for node_id, sample in data.items():
        if node_id in sample['candidates']:
            print(f'removing {node_id} from its own candidates')
            del sample['candidates'][node_id]
    if args and args['debug_maps_size']:
        data = {k: v for k, v in list(data.items())[:args['debug_maps_size']]}
    return data


def get_annotation_data_path(local: bool):
    return get_base_data_path(local) / 'annotation/annotation200instances'


def get_base_data_path(local: bool):
    return (Path.home() / "data/e-delib/kialo" if local else
            Path("/mount/projekte/e-delib/data/kialo"))

