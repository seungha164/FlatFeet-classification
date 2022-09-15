import os
import cv2

def save(src, img):
    cv2.imwrite(src, img)
def read(src):
    return cv2.imread(src)
def flip(img,mode): #mode(1:좌우, 0:상하)
    return cv2.flip(img,mode)

def preprocessing(img):
    h, w, c = img.shape
    crop = img[int(h/4):, 10:w-10]
    h1, w1, c1 = crop.shape
    space = w1-h1
    top, bottom = space-(space//2), space//2
    tmp = cv2.copyMakeBorder(crop, top, bottom, 0, 0,
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.resize(tmp, dsize=(512, 512), interpolation=cv2.INTER_AREA)

'''     main code    '''

savePath = './input/both/'
originPath = '../FlatFoot_project/dataset/' # dataset이 있는 위치
surfaceList ={'flatfeet':[], 'normal':[]}

for _class in os.listdir(originPath):
    if _class != "flatfeet" and _class != 'normal':
        continue
    if _class == "normal" :
        continue
    dataset = os.listdir(originPath+_class+'/')
    print(_class+' : '+str(len(dataset)))
    for n,_data in enumerate(dataset):
        print('['+str(n)+'] save')
        if n<280:
            finPath = savePath+'train/'+_class+'/'
        elif n<360:
            finPath = savePath+'valid/'+_class+'/'
        else:
            finPath = savePath+'test/'+_class+'/'
        
        imgL = preprocessing(read(originPath + _class+'/'+_data+'/xray/xray_L.jpg'))
        imgR = flip(preprocessing(read(originPath + _class+'/'+_data+'/xray/xray_R.jpg')),1) 
        save(finPath+'L/'+_data+'.jpg', imgL)
        save(finPath+'R/'+_data+'.jpg', imgR)