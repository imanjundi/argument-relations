import random
import re
from typing import Union, Sequence, AbstractSet

args: dict[str]


def remove_url_and_hashtags(text):
    text = re.sub(r'\[([^\]]+)\]\(http[^\)]+\)', r'\1', text)
    text = text.replace("#", "")
    return text


def sample(x: Union[Sequence, AbstractSet], max_size: int):
    if args['data_samples_seed']:
        random.seed(args['data_samples_seed'])
    return random.sample(x, min(max_size, len(x)))
