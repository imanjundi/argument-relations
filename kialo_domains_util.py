import itertools
from collections import defaultdict, Counter
import pandas as pd


def get_map2topics(path_kialo2topics):
    """Returns a dictionary of mapname with the corresponding topics"""
    topics = pd.read_csv(path_kialo2topics, sep="\t")
    topics.dropna(inplace=True)
    topics["topic_tags"] = [[w.strip().lower() for w in el.split(",")] for el in topics.topics.values]
    return dict(zip(topics.MapID, topics.topic_tags))


def get_subtopic_to_parent(maintopics):
    sutop2maintop = {}
    main2subtopic = {}
    main_tops = pd.read_csv(maintopics, sep="\t")
    for index, row in main_tops.iterrows():
        maintopic = row.maintopic
        subtopics = row.subtopics.split(",")
        subtopics = [el.strip() for el in subtopics]
        main2subtopic[maintopic] = subtopics
        for subtopic in subtopics:
            sutop2maintop[subtopic] = maintopic
    return sutop2maintop, main2subtopic


def get_map2main_topic(maps2topics, sub2maintopic):
    """
    Get the dictionary of each map to its maintopic(s)
    :param maps2topics: dictionary with each map and all its topics
    :param sub2maintopic: dictionary with all subtopics and their parent topic
    """
    maps_without_maintopic = []
    map2maintopic = defaultdict(list)
    for map_name, topic_tags in maps2topics.items():
        maintopic = None
        for tag in topic_tags:
            if tag in sub2maintopic:
                maintopic = sub2maintopic[tag]
                map2maintopic[map_name].append(maintopic)
        if not maintopic:
            maps_without_maintopic.append(map_name)
    print("%d maps do not have a main topic" % len(maps_without_maintopic))
    return map2maintopic, maps_without_maintopic


def get_unique_label(maps2maintopics):
    topic_frequency = Counter(itertools.chain(*[set(el) for el in maps2maintopics.values()]))
    map2unique_topic = {}
    for map, maintopics in maps2maintopics.items():
        if len(set(maintopics)) > 1:
            most_common = Counter(maintopics).most_common(2)
            if most_common[0][1] != most_common[1][1]:
                label = most_common[0][0]
            else:
                min_freq = 1000
                label = ""
                for topic in set(maintopics):
                    freq = topic_frequency[topic]
                    if freq < min_freq:
                        min_freq = freq
                        label = topic
            map2unique_topic[map] = label
        else:
            map2unique_topic[map] = maintopics[0]
    return map2unique_topic


def get_maps2uniquetopic(path_kialo2topics, maintopics):
    maps2topics = get_map2topics(path_kialo2topics)
    print(f'{len(maps2topics)} maps have tags')
    sub2maintopic, main2subtopic = get_subtopic_to_parent(maintopics)
    maps2maintopic, left_maps = get_map2main_topic(maps2topics, sub2maintopic)
    print("%d maps have topic tag(s)" % len(maps2maintopic))
    # print(maps2maintopic)
    map2unique = get_unique_label(maps2maintopic)
    return map2unique, (maps2topics, sub2maintopic, main2subtopic)

