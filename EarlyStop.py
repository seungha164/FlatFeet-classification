import torch
import numpy as np
class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path='./checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, path_detail):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path_detail)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            #self.save_checkpoint(val_loss, model, path_detail)
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path_detail)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path_detail):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:   # 특정 문구가 있다면 출력해주고
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path+path_detail)
        self.val_loss_min = val_loss