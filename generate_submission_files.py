import os

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import shutil
from models.RNN import RNN
from utils.misc import extract_keystroke_features
from utils.configs import configs

os.makedirs(configs.result_dir, exist_ok=True)

with open(configs.comparison_file, "r") as file:
    comps = eval(file.readline())
data = np.load(configs.test_set, allow_pickle=True).item()

model = RNN(input_size=2, hidden_size=32, output_size=32).double()
model.load_state_dict(torch.load(configs.model_dir + configs.model_filename)) # , strict=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model.to(device)

data_length = 100
dimension = 2

preproc_data = {}
i = 0
for session in list(data.keys()):
    if i % 100 == 0:
        print(str(np.round(100*(i/len(list(data.keys()))), 2)) + '% raw data preprocessed', end='\r')
    preproc_data[session] = extract_keystroke_features(data[session], data_length=data_length, dimension=dimension)
    i = i + 1

batch_size = 5000
embeddings_list = []
model.eval()
with torch.no_grad():
    all_sessions = np.array(list(preproc_data.values()))
    for i in range(int(len(all_sessions)/batch_size)):
        input_data = np.reshape(all_sessions[i*batch_size:(i+1)*batch_size], (batch_size, data_length, dimension))
        input_data = Variable(torch.from_numpy(input_data)).double().to(device)
        embeddings_list.append(model(input_data).cpu())
        print(str(np.round(100 * ((i+1) / int(len(all_sessions)/batch_size)), 2)) + '% embeddings computed', end='\r')

embeddings_list = [item for sublist in embeddings_list for item in sublist]
embeddings = {}


tmp_list = list(preproc_data.keys())
for i in range(len(tmp_list)):
    embeddings[tmp_list[i]] = ''
embeddings.update(zip(embeddings, embeddings_list))


distances = {}
i = 0
for comp in comps:
    distance = nn.functional.pairwise_distance(embeddings[comp[0]], embeddings[comp[1]]).item()
    distances[str(comp)] = distance
    if i % 1000 == 0:
        print(str(np.round(100 * ((i + 1) / len(comps)), 2)) + '% distances computed', end='\r')
    i = i + 1

distances_list = list(distances.values())
max_dist = max(distances_list)
distances_list = [1-(x/max_dist) for x in distances_list]

with open(configs.result_filename, "w") as file:
    file.write(str(distances_list))

shutil.make_archive(configs.experiment_name, 'zip', configs.result_dir)
