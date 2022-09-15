import os,time
import numpy as np
import torch,gc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from EarlyStop import EarlyStopping         # early stop을 위한 클래스
import torchvision.transforms as T

from Dataset import getDataset
from model import getModel

def calc_accuracy(true,pred):
    pred = F.softmax(pred, dim = 1)
    true = torch.zeros(pred.shape[0], pred.shape[1]).scatter_(1, true.unsqueeze(1), 1.)
    acc = (true.argmax(-1) == pred.argmax(-1)).float().detach().numpy()
    acc = float((100 * acc.sum()) / len(acc))
    return round(acc, 4)

def training_loop(n_epochs, patience,n_inputs=2):
    train_loss, val_loss = [], []
    train_accuracy, val_accuracy = [], []
    
    # early_stopping object의 초기화
    early_stopping = EarlyStopping(patience = patience, verbose = True,path=MODEL_PATH)
    for epoch in range(1,n_epochs+1):
        start = time.time()
        #Epoch Loss & Accuracy, Val Loss & Accuracy
        train_epoch_loss, train_epoch_accuracy= [],[]
        val_epoch_loss, val_epoch_accuracy = [], []
        ###################
        # train the model #
        ###################
        model.train() 
        for trainData in train_data_loader:
            # reset Grads
            optimizer.zero_grad()
            if n_inputs == 1:
                img, label = trainData
                img, label = img.to(device), label.to(device)
                output = model(img)                 # 1. Forward
            elif n_inputs == 2:
                imgL, imgR, label = trainData
                imgL, imgR, label = imgL.to(device), imgR.to(device), label.to(device)
                output = model(imgL,imgR)           # 1. Forward
            else:
                imgL,imgR,imgF,label = trainData
                imgL,imgR,imgF,label = imgL.to(device),imgR.to(device),imgF.to(device), label.to(device)
                output = model(imgL,imgR,imgF)      # 1. Forward
        
            # 2. Calculate Accuracy            
            acc = calc_accuracy(label.cpu(),output.cpu())
            #break
            # 3. loss 계산 & Backward. weights 업데이트
            loss = loss_fn(output,label)
            loss.backward()
            optimizer.step()
            #Append loss & acc
            loss_val = loss.item()
            train_epoch_loss.append(loss_val)
            train_epoch_accuracy.append(acc)

        train_epoch_loss, train_epoch_accuracy = np.mean(train_epoch_loss), np.mean(train_epoch_accuracy)
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)

        lr_scheduler.step()
        ###################
        # valid the model #
        ###################
        model.eval() 
        with torch.no_grad():
            for validData in valid_data_loader:
                if n_inputs == 1:
                    img, label = validData
                    img, label = img.to(device), label.to(device)
                    pred = model(img)                            # 1. Forward
                elif n_inputs == 2:
                    imgL, imgR, label = validData
                    imgL, imgR, label = imgL.to(device), imgR.to(device), label.to(device)
                    pred = model(imgL,imgR)                      # 1. Forward
                else:
                    imgL,imgR,imgF,label = validData
                    imgL,imgR,imgF,label = imgL.to(device),imgR.to(device),imgF.to(device), label.to(device)
                    pred = model(imgL,imgR,imgF)                 # 1. Forward
                
                acc = calc_accuracy(label.cpu(), pred.cpu())        #Calculate Acc
                loss = loss_fn(pred, label)                         #Calculate Loss
                loss_value = loss.item()
                val_epoch_loss.append(loss_value)
                val_epoch_accuracy.append(acc)
            val_epoch_loss, val_epoch_accuracy = np.mean(val_epoch_loss), np.mean(val_epoch_accuracy)
            val_loss.append(val_epoch_loss)
            val_accuracy.append(val_epoch_accuracy)
        end = time.time()
        #Print Epoch Statistics
        print("** Epoch {} ** - Epoch Time {}s".format(epoch, int(end-start)))
        print("Train Loss = {}".format(round(train_epoch_loss, 4)))
        print("Train Accuracy = {} % \n".format(train_epoch_accuracy))
        print("Val Loss = {}".format(round(val_epoch_loss, 4)))
        print("Val Accuracy = {} % \n".format(val_epoch_accuracy))
        early_stopping(val_epoch_loss, model,path_detail=str(epoch)+'.pt')
        if early_stopping.early_stop:
            print("Early stopping")
            break
    model.load_state_dict(torch.load(MODEL_PATH))
    return  model, train_loss, val_loss, train_accuracy, val_accuracy

def Visualizing_Loss_EarlyStoppingCheckpoint(train_loss, valid_loss, train_acc, valid_acc):
    train_acc, valid_acc = [i/100 for i in train_acc], [i/100 for i in valid_acc]
    print('=>',train_loss,valid_loss,train_acc,valid_acc)
    # 훈련이 진행되는 과정에 따라 loss를 시각화
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')
    #plt.plot(range(1,len(train_acc)+1),train_acc, label='Training Acc')
    #plt.plot(range(1,len(valid_acc)+1),valid_acc,label='Validation Acc')
    # validation loss의 최저값 지점을 찾기
    minposs = valid_loss.index(min(valid_loss))+1
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 1) # 일정한 scale
    plt.xlim(0, len(valid_loss)+1) # 일정한 scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def init():
    gc.collect()
    torch.cuda.empty_cache()


transform_test = T.Compose([T.ToTensor()])
transform_train = T.Compose([
    T.ToTensor(),
    T.ColorJitter(hue=.05, saturation=.05),
    T.RandomRotation(0.5)

])
    
#############################################################
##                        MAIN CODE                        ##
#############################################################
if __name__ == "__main__":
    init()
    N_INPUT = 2
    MODEL = 'M5'
    DIR_TRAIN = "./input/train/"
    DIR_VALID = "./input/valid/"
    MODEL_PATH = './MODEL/M5/epoch_'

    DatasetList = {1:'SingleDataset',2:'DualDataset',3:'TripleDataset'}
    train_dataset = getDataset(DatasetList[N_INPUT], DIR_TRAIN, transform_train)
    valid_dataset = getDataset(DatasetList[N_INPUT], DIR_VALID, transform_test)
    train_data_loader = DataLoader(
        dataset = train_dataset,
        batch_size = 10,
        shuffle=True,
        num_workers = 4,
    )
    valid_data_loader = DataLoader(
        dataset = valid_dataset,
        batch_size = 10,
        shuffle=True,
        num_workers = 4,
    )
    
    # 1. check CUDA or CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    # 2. check MODEL
    model = getModel(MODEL)
    model = model.to(device)
    # 3. check TRAIN_PARAMETER
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    #lr_scheduler = get_cosine_schedule_with_warmup(optimizer,5,0.3,0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.75)
    
    loss_fn=nn.CrossEntropyLoss()           # 손실함수 set
    # 4. early stopping patience;
    patience = 7   # validation loss가 개선된 마지막 시간 이후로 얼마나 기다릴지 지정
    model, train_loss, valid_loss, train_acc, valid_acc = training_loop(n_epochs=300,patience=patience,n_inputs=N_INPUT)
    Visualizing_Loss_EarlyStoppingCheckpoint(train_loss, valid_loss, train_acc, valid_acc)
    print('training finish')
