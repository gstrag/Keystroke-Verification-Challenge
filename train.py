import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import roc_curve
import torch.nn as nn
import time
from models.RNN import RNN
from utils.losses import contrastive_loss
from utils.misc import PrepareData
from utils.configs import configs
import os

os.makedirs(configs.base_dir, exist_ok=True)
os.makedirs(configs.log_dir, exist_ok=True)
os.makedirs(configs.model_dir, exist_ok=True)

data = np.load(configs.dev_set, allow_pickle=True).item()


dev_set_users = list(data.keys())
random.shuffle(dev_set_users)
train_users = dev_set_users[:int(len(dev_set_users)*configs.train_val_division)]
val_users = dev_set_users[int(len(dev_set_users)*configs.train_val_division):]


train_data = copy.deepcopy(data)
for user in list(data.keys()):
    if user not in train_users:
        del train_data[user]
val_data = copy.deepcopy(data)
for user in list(data.keys()):
    if user not in val_users:
        del val_data[user]
del data



ds_t = PrepareData(train_data, sequence_length=configs.sequence_length, samples_considered_per_epoch=configs.batches_per_epoch_train*configs.batch_size_train)
ds_v = PrepareData(val_data, sequence_length=configs.sequence_length, samples_considered_per_epoch=configs.batches_per_epoch_val*configs.batch_size_val)

train_dataloader = DataLoader(ds_t, batch_size=configs.batch_size_train)
val_dataloader = DataLoader(ds_v, batch_size=configs.batch_size_val)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = RNN(input_size=2, hidden_size=32, output_size=32).double()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))



def train_one_epoch():
    running_loss = 0.
    batch_eers = []
    batch_losses = []
    for i, (input_data, labels) in enumerate(train_dataloader, 0):
        optimizer.zero_grad()
        input_data[0] = Variable(input_data[0]).double()
        input_data[1] = Variable(input_data[1]).double()
        input_data[0] = input_data[0].to(device)
        input_data[1] = input_data[1].to(device)
        pred1 = model(input_data[0])
        pred2 = model(input_data[1])
        loss = contrastive_loss(pred1, pred2, labels.to(torch.int64).double().to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        pred1 = pred1.cpu().detach()
        pred2 = pred2.cpu().detach()
        dists = nn.functional.pairwise_distance(pred1, pred2)
        fpr, tpr, threshold = roc_curve(labels, dists, pos_label=1)
        fnr = 1 - tpr
        EER1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        EER2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
        batch_eers.append(np.mean([EER1, EER2])*100)
        batch_losses.append(running_loss)
    return np.sum(batch_losses), np.mean(batch_eers)



loss_t_list, eer_t_list = [], []
loss_v_list, eer_v_list = [], []

best_v_eer = 100.
for epoch in range(configs.num_epochs):
    start = time.time()
    print('EPOCH:', epoch)
    model.train()
    epoch_loss, epoch_eer = train_one_epoch()
    loss_t_list.append(epoch_loss)
    eer_t_list.append(epoch_eer)
    model.eval()
    running_loss_v = 0.
    with torch.no_grad():
        batch_eers_v = []
        batch_losses_v = []
        for i, (input_data, labels) in enumerate(val_dataloader, 0):
            input_data[0] = Variable(input_data[0]).double()
            input_data[1] = Variable(input_data[1]).double()
            input_data[0] = input_data[0].to(device)
            input_data[1] = input_data[1].to(device)
            pred1 = model(input_data[0])
            pred2 = model(input_data[1])            # criterion = torch.jit.script(nn.BCELoss())
            loss_v = contrastive_loss(pred1, pred2, labels.to(torch.int64).double().to(device))
            running_loss_v += loss_v.item()
            pred1 = pred1.cpu().detach()
            pred2 = pred2.cpu().detach()
            dists = nn.functional.pairwise_distance(pred1, pred2)
            fpr, tpr, threshold = roc_curve(labels, dists, pos_label=1)
            fnr = 1 - tpr
            EER1_v = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            EER2_v = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
            batch_eers_v.append(np.mean([EER1_v, EER2_v])*100)
            batch_losses_v.append(running_loss_v)
    eer_v_list.append(np.mean(batch_eers_v))
    loss_v_list.append(np.sum(batch_losses_v))
    print("Epoch Loss on Training Set: " + str(epoch_loss))
    print("Epoch Loss on Validation Set: " + str(np.sum(batch_losses_v)))
    print("Epoch EER [%] on Training Set: "+ str(epoch_eer))
    print("Epoch EER [%] on Validation Set: " + str(np.mean(batch_eers_v)))
    if eer_v_list[-1] < best_v_eer:
        torch.save(model.state_dict(), configs.model_dir + configs.model_filename)
        print("New Best Epoch EER [%] on Validation Set: " + str(eer_v_list[-1]))
        best_v_eer = eer_v_list[-1]
    log_list = [loss_t_list, loss_v_list, eer_t_list, eer_v_list]
    with open(configs.training_log_filename, "w") as output:
        output.write(str(log_list))
    end = time.time()
    print('Time for last epoch [min]: ' + str(np.round((end-start)/60, 2)))


