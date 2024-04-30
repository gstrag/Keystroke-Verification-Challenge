import numpy as np
import random

class PrepareData:
    def __init__(self, dataset, sequence_length, samples_considered_per_epoch):
        self.data = dataset
        self.len = samples_considered_per_epoch
        self.sequence_length = sequence_length

    def __getitem__(self, index):
        user_idx = random.choice(list(self.data.keys()))
        session_idx = random.choice(list(self.data[user_idx].keys()))
        session_1 = self.data[user_idx][session_idx]
        session_1 = np.concatenate((session_1, np.zeros((self.sequence_length, np.shape(session_1)[1]))))[:self.sequence_length]
        diff_1 = np.reshape((session_1[:, 1] - session_1[:, 0]) / 1E3, (np.shape(session_1)[0], 1))
        ascii_1 = np.reshape(session_1[:, 2] / 256, (np.shape(session_1)[0], 1))
        session_1_processed = np.concatenate((diff_1, ascii_1), axis=1)

        label = random.choice([0, 1])
        if label == 0:
            session_idx_2 = random.choice([x for x in list(self.data[user_idx].keys()) if x != session_idx])
            session_2 = self.data[user_idx][session_idx_2]
        else:
            user_idx_2 = random.choice([x for x in list(self.data.keys()) if x != user_idx])
            session_idx_2 = random.choice(list(self.data[user_idx_2].keys()))
            session_2 = self.data[user_idx_2][session_idx_2]
        session_2 = np.concatenate((session_2, np.zeros((self.sequence_length, np.shape(session_2)[1]))))[:self.sequence_length]
        diff_2 = np.reshape((session_2[:, 1] - session_2[:, 0]) / 1E3, (np.shape(session_2)[0], 1))
        ascii_2 = np.reshape(session_2[:, 2] / 256, (np.shape(session_2)[0], 1))
        session_2_processed = np.concatenate((diff_2, ascii_2), axis=1)

        return (session_1_processed, session_2_processed), label

    def __len__(self):
        return self.len


def extract_keystroke_features(session_key, data_length = 100, dimension = 2):
    diff = np.reshape((session_key[:, 1] - session_key[:, 0]) / 1E3, (np.shape(session_key)[0], 1))
    ascii = np.reshape(session_key[:, 2] / 256, (np.shape(session_key)[0], 1))
    keys_features = np.concatenate((diff, ascii), axis = 1)
    output = np.concatenate((keys_features, np.zeros((data_length, dimension))))[:data_length]
    return output