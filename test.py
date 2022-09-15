import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from confusionMatirx import draw_confusionMatrix
from model import getModel
from Dataset import getDataset


def test(n_inputs=2):
    model.to(device)
    true, pred = [], []
    className = {0:'normal',1:'flatfeet'}
    model.eval()
    with torch.no_grad():
        accuracy, total = 0.0, 0.0
        for total,testData in enumerate(test_data_loader,1):
            #total+=1
            if n_inputs == 1:
                img, label = testData
                img, label = img.to(device), label.to(device)
                output = model(img)
            elif n_inputs == 2:
                imgL, imgR, label = testData
                imgL, imgR, label = imgL.to(device), imgR.to(device), label.to(device)
                output = model(imgL,imgR)
            else:
                imgL, imgR, imgF, label = testData
                imgL, imgR, imgF, label = imgL.to(device), imgR.to(device), imgF, label.to(device)
                output = model(imgL,imgR,imgF)
            probabilities = torch.exp(output)
            equality = (label.data == probabilities.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
            labelsT = label.item()
            prT = probabilities.max(dim=1)[1].item()
            print('({}) : {} => {}      [accuracy : {}%]'.format(total,className[labelsT],className[prT],(accuracy/total)))
            # confusion-matrix data 저장
            true.append(int(label.item()))
            pred.append(int((probabilities.max(dim=1)[1][0]).item()))
        print("Test Accuracy: {}%".format(accuracy/total*100))
    return true,pred,(accuracy/total*100)


#######################################
##             main code             ##
#######################################
N_INPUT = 2
MODEL = 'M5'
DIR_TEST = "./input/test/"
DIR_MODEL = './MODEL/M5/M5_best.pt'    #'C:/pesPlanus/model/18.pt'

DatasetList = {1:'SingleDataset',2:'DualDataset',3:'TripleDataset'}
transform_test = T.Compose([T.ToTensor()])
if __name__ == "__main__":
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    # 1. test data 불러오기
    test_dataset = getDataset(DatasetList[N_INPUT],DIR_TEST,transform_test)
    test_data_loader = DataLoader(
        dataset = test_dataset,
        batch_size = 1,
        shuffle=False,
        num_workers = 4,
    )
    # 2. 모델 불러오기
    model = getModel(MODEL) 
    model.load_state_dict(torch.load(DIR_MODEL))
    model.to(device)
    # 3. test
    confusion_data = test(n_inputs=N_INPUT)
    # 4. confusion matrix
    draw_confusionMatrix(['normal','flat feet'],confusion_data, _normalize=True)


