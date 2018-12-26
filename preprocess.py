import numpy as np
import re
import json

FIXED_PARAMETERS = {
    "train_mnli": "../data/multinli_0.9/multinli_0.9_train.jsonl",
    "dev_mnli": "../data/multinli_0.9/multinli_0.9_dev_matched.jsonl",
    "dev_mismatched": "../data/multinli_0.9/multinli_0.9_dev_mismatched.jsonl",
    "test_mnli": "../data/multinli_0.9/multinli_0.9_dev_matched.jsonl",
    "test_mismatched": "../data/multinli_0.9/multinli_0.9_dev_mismatched.jsonl",
    "train_snli": "../data/snli_1.0/snli_1.0_train.jsonl",
    "dev_snli": "../data/snli_1.0/snli_1.0_dev.jsonl",
    "test_snli": "../data/snli_1.0/snli_1.0_test.jsonl"
}

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}

def clean_str(string):
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " ( ", string)
  string = re.sub(r"\)", " ) ", string)
  string = re.sub(r"\?", " ? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower()

class Data():
    def __init__(self, max_len=0):
        self.s1s, self.s2s, self.labels, self.features = [], [], [], []
        self.index, self.max_len = 0, max_len
        self.src_s1s, self.src_s2s, self.src_labels, self.src_features = [], [], [], []
        self.src_index = 0

    def open_file(self):
        pass

    def is_available(self):
        if self.index < self.data_size:
            return True
        else:
            return False

    def reset_index(self):
        self.index = 0

    def next(self):
        if (self.is_available()):
            self.index += 1
            return self.data[self.index - 1]
        else:
            return

    def gen_data(self, word2idx, max_len):
        s1_mats, s2_mats = [], []
        s1_seq_len, s2_seq_len = [], []

        for i in range(self.data_size):
            s1 = self.s1s[i]
            s2 = self.s2s[i]
            s1_seq_len.append(len(s1))
            s2_seq_len.append(len(s2))

            # [1, d0, s]
            s1_idx = []
            for w in s1:
                if w in word2idx:
                    s1_idx.append(word2idx[w])
                else:
                    print('word not in vocab')
            for _ in range(max_len-len(s1)):
                s1_idx.append(0)
            s2_idx = []
            for w in s2:
                if w in word2idx:
                    s2_idx.append(word2idx[w])
                else:
                    print('word not in vocab')
            for _ in range(max_len-len(s2)):
                s2_idx.append(0)
            s1_mats.append(s1_idx)
            s2_mats.append(s2_idx)

        # [batch_size, d0, s]
        total_s1s = np.asarray(s1_mats)
        total_s2s = np.asarray(s2_mats)

        return total_s1s, total_s2s#, s1_seq_len, s2_seq_len

    '''
    def next_batch(self, total_s1s, total_s2s, batch_size):
        batch_size = min(self.data_size - self.index, batch_size)

        # [batch_size, d0, s]
        batch_s1s = total_s1s[self.index:self.index + batch_size]
        batch_s2s = total_s2s[self.index:self.index + batch_size]
        batch_labels = self.labels[self.index:self.index + batch_size]
        batch_features = self.features[self.index:self.index + batch_size] #not useful anymore

        self.index += batch_size

        return batch_s1s, batch_s2s, batch_labels, batch_features#, s1_seq_len, s2_seq_len
    '''

    def next_batch(self, total_s1s, total_s2s, labels, batch_size):
        batch_size = min(self.data_size - self.index, batch_size)

        # [batch_size, d0, s]
        batch_s1s = total_s1s[self.index:self.index + batch_size]
        batch_s2s = total_s2s[self.index:self.index + batch_size]
        batch_labels = labels[self.index:self.index + batch_size]
        batch_features = self.features[self.index:self.index + batch_size]

        self.index += batch_size

        return batch_s1s, batch_s2s, batch_labels, batch_features#, s1_seq_len, s2_seq_len

class SNLI(Data):
    def open_file(self, mode, domain, genre, parsing_method="normal"):
        len_thres = 111
        src_input_name = FIXED_PARAMETERS[mode + "_snli"]
        tgt_input_name = FIXED_PARAMETERS[mode + "_mnli"]
        if domain == 'tgt':
            with open(tgt_input_name, "r") as f:
                for line in f:
                    loaded_example = json.loads(line)
                    if loaded_example["gold_label"] not in LABEL_MAP:
                        continue
                    if loaded_example["genre"] != genre:
                        continue
                    label = LABEL_MAP[loaded_example["gold_label"]]

                    s1 = clean_str(loaded_example["sentence1"]).strip().split(' ')
                    s2 = clean_str(loaded_example["sentence2"]).strip().split(' ')
                    if len(s1) >= len_thres:
                        s1 = s1[:len_thres + 1]
                    if len(s2) >= len_thres:
                        s2 = s2[:len_thres + 1]

                    # bleu_score = nltk.translate.bleu_score.sentence_bleu(s1, s2)
                    # sentence_bleu(s1, s2, smoothing_function=nltk.translate.bleu_score.SmoothingFunction.method1)

                    self.s1s.append(s1)
                    self.s2s.append(s2)
                    self.labels.append(label)
                    self.features.append([len(s1), len(s2)])

                    local_max_len = max(len(s1), len(s2))
                    if local_max_len > self.max_len:
                        self.max_len = local_max_len

            self.data_size = len(self.s1s)
            self.num_features = len(self.features[0])
        else:
            with open(src_input_name, "r") as f:
                for line in f:
                    loaded_example = json.loads(line)
                    if loaded_example["gold_label"] not in LABEL_MAP:
                        continue
                    label = LABEL_MAP[loaded_example["gold_label"]]

                    s1 = clean_str(loaded_example["sentence1"]).strip().split(' ')
                    s2 = clean_str(loaded_example["sentence2"]).strip().split(' ')
                    if len(s1) >= len_thres:
                        s1 = s1[:len_thres + 1]
                    if len(s2) >= len_thres:
                        s2 = s2[:len_thres + 1]

                    # bleu_score = nltk.translate.bleu_score.sentence_bleu(s1, s2)
                    # sentence_bleu(s1, s2, smoothing_function=nltk.translate.bleu_score.SmoothingFunction.method1)

                    self.s1s.append(s1)
                    self.s2s.append(s2)
                    self.labels.append(label)
                    self.features.append([len(s1), len(s2)])

                    local_max_len = max(len(s1), len(s2))
                    if local_max_len > self.max_len:
                        self.max_len = local_max_len

            self.data_size = len(self.s1s)
            self.num_features = len(self.features[0])
