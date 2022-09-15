# FlatFeet classification
FlatFeet x-ray image classification project
### 1. Input Image Preprocessing
 <img src="https://user-images.githubusercontent.com/87134443/190317108-bf064471-a59e-4015-8cfe-09933a446341.jpg" width="600" height="200"/>

### 2. Train
- train image : 560
- valid image : 80
- class : 2 (normal / flatfeet)
- optimizer : Adam
- epoch : 300 (Early stopping)
- learning rate : 0.001
- batch size : 8
- model : M5(custom dual input model)
 ![image](https://user-images.githubusercontent.com/87134443/190319163-dc0cef83-ea7f-4b54-81ec-17709dc8f03a.png)
### 3. Test Acc & Confusion Matrix
Test Image : 80 (normal 40 + flatfeet 40)
- test accuracy : 71.25%
- confusion matrix
 <img src="https://user-images.githubusercontent.com/87134443/190314993-27daa09b-d934-487f-9aec-a23ad0cc6bed.png" width="400" height="300"/>
