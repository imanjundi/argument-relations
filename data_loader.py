import random
import math

import sentence_transformers


class SameMapPerBatchDataLoader(sentence_transformers.datasets.NoDuplicatesDataLoader):

    def __init__(self, train_examples, batch_size):
        super(SameMapPerBatchDataLoader, self).__init__(train_examples, batch_size)
        for x in train_examples:
            random.shuffle(x)
        self.data_pointers = [0] * len(train_examples)

    def __iter__(self):
        for _ in range(self.__len__()):
            batch = []
            texts_in_batch = set()

            while len(batch) < self.batch_size:
                example = self.train_examples[self.data_pointer][self.data_pointers[self.data_pointer]]

                valid_example = True
                for text in example.texts:
                    if text.strip().lower() in texts_in_batch:
                        valid_example = False
                        break

                if valid_example:
                    batch.append(example)
                    for text in example.texts:
                        texts_in_batch.add(text.strip().lower())

                self.data_pointers[self.data_pointer] += 1
                if self.data_pointers[self.data_pointer] >= len(self.train_examples[self.data_pointer]):
                    self.data_pointers[self.data_pointer] = 0
                    random.shuffle(self.train_examples[self.data_pointer])
                    self.progress_to_next_map()

            self.progress_to_next_map()

            yield self.collate_fn(batch) if self.collate_fn is not None else batch

    def progress_to_next_map(self):
        self.data_pointer += 1
        if self.data_pointer >= len(self.train_examples):
            self.data_pointer = 0
            random.shuffle(self.train_examples)

    def __len__(self):
        return math.floor(sum([len(x) for x in self.train_examples]) / self.batch_size)


# based on sentence_transformers.datasets.NoDuplicatesDataLoader
def validate_for_no_duplicates_batch(train_examples, batch_size, strict=True):
    texts_in_batch = set()
    possible_batch_size = 0

    for example in train_examples:
        valid_example = True
        for text in example.texts:
            if text.strip().lower() in texts_in_batch:
                valid_example = False
                break

        if valid_example:
            possible_batch_size += 1
            for text in example.texts:
                texts_in_batch.add(text.strip().lower())

    if strict:
        assert possible_batch_size >= batch_size, \
            f"not enough samples: {possible_batch_size=} < {batch_size=}"

    return possible_batch_size
