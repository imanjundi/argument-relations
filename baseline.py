import logging

import os
from pathlib import Path

from encode_nodes import MapEncoder
from argumentMap import KialoMap, DeliberatoriumMap
from eval_util import evaluate_map
from sentence_transformers import LoggingHandler

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def deliberatorium_baseline():
    data_path = Path.home() / "data/e-delib/deliberatorium/maps/english_maps"
    maps = os.listdir(data_path)

    encoder_mulitlingual = MapEncoder(max_seq_len=128, sbert_model_identifier="all-mpnet-base-v2",
                                      normalize_embeddings=True, use_descriptions=False)

    for map in maps:
        argument_map = DeliberatoriumMap("%s/%s" % (str(data_path), map))
        evaluate_map(encoder_mulitlingual, argument_map, {"issue", "idea"})


def kialo_baseline():
    data_path = Path.home() / "data/e-delib/deliberatorium/maps/kialo_maps"
    maps = os.listdir(data_path)
    encoder_mulitlingual = MapEncoder(max_seq_len=128, sbert_model_identifier="all-mpnet-base-v2",
                                      normalize_embeddings=True, use_descriptions=False)
    for map in maps:
        print(map)
        path = "%s/%s" % (data_path, map)
        argument_map = KialoMap(path)
        evaluate_map(encoder_mulitlingual, argument_map, {"Pro"})


if __name__ == '__main__':
    kialo_baseline()
