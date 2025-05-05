import torch
import torch.nn as nn
import torch.optim as optim
import copy



import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
import copy 
import pickle
from torch.optim.lr_scheduler import ExponentialLR


class SegmentationTrainer2:
    def __init__(self, model1, train_dataloader, val_dataloader, num_classes, num_epochs, device, name='exp1',case=4,model2=None, lr=None, criterion=None ):
        self.model1 = model1.to(device)
        self.model2 = model2.to(device) if model2 else None

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.device = device

        self.name=name
        self.step=1/self.num_epochs


        self.best_loss = float('inf')
        self.best_alpha = 1
        self.bestbeta = 1
        self.lr=lr if lr else 0.001 
        self.alpha_list=[]
        self.beta_list=[]

        
        
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        

        self.case=case

        #Modes for training
        self.modes=['model1'] if self.case==2 else ['model1', 'model2', 'combined' ]



        if self.case==1: #Alpha and beta optimized linearly 

            self.optimizer = optim.Adam([
                {'params': model1.parameters()},
                {'params': model2.parameters()}
            ], lr=self.lr)

            self.alpha=1
            self.alpha_list.append(self.alpha)


        elif self.case==2: #Finetune model 1 and that is it 
            
            self.optimizer = optim.Adam(model1.parameters(), lr=self.lr)
            # Initialize scheduler
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)


        elif self.case==3: #Freeze 1 and train 2 with linear alpha and beta 

            self.alpha=1
            self.alpha_list.append(self.alpha)

            for param in self.model1.parameters():
                param.requires_grad = False
            
            self.optimizer= optim.Adam(model2.parameters(), lr=self.lr)
                

        elif self.case==4: #Alpha and beta part of optimizer

            self.alpha = torch.nn.Parameter(torch.tensor(0.5, device=device, requires_grad=True))
            self.beta = torch.nn.Parameter(torch.tensor(0.5, device=device, requires_grad=True))
            #self.alpha=torch.sigmoid(self.alpha)

            #self.alpha_list.append(self.alpha.item())
        
            self.optimizer = optim.Adam([
                {'params': model1.parameters()},
                {'params': model2.parameters()},
                {'params': [self.alpha]}
            ], lr=self.lr)
        
        elif self.case==5: #Alpha and beta part of optimizer, both. not add up to 1 
            self.alpha = torch.nn.Parameter(torch.tensor(0.5, device=device, requires_grad=True))
            self.beta = torch.nn.Parameter(torch.tensor(0.5, device=device, requires_grad=True))
            self.optimizer = optim.Adam([
                {'params': model1.parameters()},
                {'params': model2.parameters()},
                {'params': [self.alpha]},
                {'params': [self.beta]}
            ], lr=self.lr)
        
     #   self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
            #self.alpha_list.append(self.alpha)

        

    def print_statement(self):
        if self.case==1:
            print('\n Training model 1. Training model 2')
            print(f'\n Alpha and Beta are linear. Step is {self.step:2f}')
            
        elif self.case==2:
            print('Finetuning model 1. No model 2 exist')

        elif self.case==3:
            print('\n Freezing model 1. Training model 2')
        elif self.case==4  or self.case==5:
            print('\n Training model 1. Training model 2')
            print('\n Alpha and Beta included in the optimizer')

                

    def calculate_beta(self):
        return 1-self.alpha

    def train_one_epoch(self,phase, dice_coefficient): 
        epoch_loss=0
        epoch_loss1=0
        epoch_loss2=0
        dice = torch.zeros(self.num_classes, device=self.device) 
        dice1 = torch.zeros(self.num_classes, device=self.device)
        dice2 = torch.zeros(self.num_classes, device=self.device) 

        dataloader = self.train_dataloader if phase == 'train' else self.val_dataloader
        dataset = dataloader.dataset

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)

            with torch.set_grad_enabled(phase == 'train'):


                if self. case!=2: #2 models a b linear:  
                    outputs1 = self.model1(images)['out']
                    loss1 = self.criterion(outputs1, labels.long()) 

                    outputs2 = self.model2(images)['out'] 
                    loss2 = self.criterion(outputs2, labels.long()) 

                    if self.case==4:
                        self.norm_alpha = torch.sigmoid(self.alpha)
                        self.norm_beta = 1 - self.norm_alpha
                     #   b=1-a
                        self.alpha_list.append(self.norm_alpha.item())
                        self.beta_list.append(self.norm_beta.item())

                    elif self.case==5:
                        self.norm_alpha = torch.sigmoid(self.alpha)
                        self.norm_beta = torch.sigmoid(self.beta)

                        self.alpha_list.append(self.norm_alpha.item())
                        self.beta_list.append(self.norm_beta.item())               
                    else:
                        self.norm_alpha=self.alpha
                        self.norm_beta=1-self.alpha
                    


                    combined = self.norm_alpha * outputs1 + self.norm_beta*outputs2 
                    loss = self.criterion(combined , labels.long())

                else: #finetune 1 
                    outputs1 = self.model1(images)['out']
                   # print(f'outputs: {outputs1.shape}')

                    #print("labels range:", labels.min().item(), labels.max().item())

                    loss1=loss=self.criterion(outputs1, labels.long()) 

                    #combined=outputs1

                if phase=='train':
                    # Backward pass and optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        
                   
            #Update metrics
            epoch_loss += loss.item() * images.size(0) 
            epoch_loss1 += loss1.item() * images.size(0) 
            epoch_loss2 += loss2.item() * images.size(0) if self.case!=2 else torch.tensor(0.0)

            dice1 += dice_coefficient(outputs1, labels, self.num_classes) #This dice is going to be added X times. X is len(dataloader)
            dice2 += dice_coefficient(outputs2, labels, self.num_classes) if self.case!=2 else torch.zeros(self.num_classes, device=self.device) 
            dice += dice_coefficient(combined, labels, self.num_classes) if self.case!=2 else torch.zeros(self.num_classes, device=self.device) 

        #Now we it has run through all images, we need to divide by lens 
        epoch_loss /= len(dataset)
        epoch_loss1 /= len(dataset)
        epoch_loss2 /= len(dataset)
        
        dice1/=len(dataloader)
        dice2/=len(dataloader)
        dice/=len(dataloader)

        self.scheduler.step()


        return epoch_loss,epoch_loss1,epoch_loss2,dice1,dice2,dice


    def update_metrics(self,phase, epoch_loss,epoch_loss1,epoch_loss2,dice1,dice2,dice):

        #Update loss
        self.epoch_loss[phase]['model1'].append(epoch_loss1)
        if self.case!=2:
            self.epoch_loss[phase]['model2'].append(epoch_loss2)
            self.epoch_loss[phase]['combined'].append(epoch_loss)

        #Update Dice
        for c in range(self.num_classes):
            self.dice_scores[phase]['model1'][c].append(dice1[c].item())
            if self.case!=2:
                self.dice_scores[phase]['model2'][c].append(dice2[c].item()) 
                self.dice_scores[phase]['combined'][c].append(dice[c].item())

    

    def set_model_mode(self, phase):
        if phase == 'train' :
            if self.case==1 or self.case==4 or self.case==5:
                self.model1.train()
                self.model2.train()
            elif self.case==2:
                self.model1.train()
            elif self.case==3:
                self.model1.eval()
                self.model2.train()

        else:
            if self.case!=2:
                self.model1.eval()
                self.model2.eval()
            else:
                self.model1.eval()



    def train(self, dice_coefficient):

        phases=['train', 'val']

        self.dice_scores={phase: {mode: {c: [] for c in range(self.num_classes)} for mode in self.modes} for phase in phases}
        
        self.epoch_loss = {phase: {mode: [] for mode in self.modes} for phase in ['train', 'val']}
    
        for epoch in range(self.num_epochs):
            #print(f'Epoch {epoch}, alpha {self.alpha:.4f}')
            #Start of epoch
            for phase in phases:

                self.set_model_mode(phase)

                epoch_loss,epoch_loss1,epoch_loss2,dice1,dice2,dice= self.train_one_epoch(phase, dice_coefficient)

                if self.case!=2:

                    print(f"Epoch [{epoch + 1}/{self.num_epochs}], {phase} Loss: {epoch_loss:.4f}, Alpha: {self.norm_alpha:.4f}, Beta: {self.norm_beta:.4f}")

                else: 
                    print(f"Epoch [{epoch + 1}/{self.num_epochs}], {phase} Loss: {epoch_loss:.4f}")


                self.update_metrics(phase, epoch_loss,epoch_loss1,epoch_loss2,dice1,dice2,dice )

                #Update alpha and beta 

                if phase=='train' and (self.case==1 or self.case==3):
                    self.alpha-=self.step
                    self.alpha_list.append(self.alpha)



                if phase=='val' and epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.best_alpha = self.alpha if self.case!=2 else None 
                    self.best_beta= self.beta if self.case!=2 else None 
                    #print(f'Best alpha {self.best_alpha} at epoch {epoch}')
                    self.best_model_wts1 = copy.deepcopy(self.model1.state_dict())
                    self.best_model_wts_path1 = f"{self.name}_model1_best_model_epoch_{epoch}.pth"
                    if self.case!=2:
                        self.best_model_wts2 = copy.deepcopy(self.model2.state_dict())
                        self.best_model_wts_path2 = f"{self.name}_model2_best_model_epoch_{epoch}.pth"

                if epoch==(self.num_epochs-1): #Save model of last epoch to compare 

                    self.last_model_wts1 = copy.deepcopy(self.model1.state_dict())
                    self.last_model_wts_path1 = f"{self.name}_model1_last_model_epoch_{epoch}.pth"
                    if self.case!=2:
                    
                        self.last_model_wts2 = copy.deepcopy(self.model2.state_dict())
                        self.last_model_wts_path2 = f"{self.name}_model2_last_model_epoch_{epoch}.pth"
                        self.last_alpha=copy.deepcopy(self.alpha)
                        self.last_beta=copy.deepcopy(self.beta)
            #End of epoch

            
        
        print("Training complete!")

    def save_best_models(self, path):
        torch.save(self.best_model_wts1, path+self.best_model_wts_path1)
        torch.save(self.last_model_wts1, path+self.last_model_wts_path1)
        if self.model2:
            torch.save(self.best_model_wts2, path+self.best_model_wts_path2)
            torch.save(self.last_model_wts2, path+self.last_model_wts_path2)
            torch.save(self.best_alpha,path+ '/best_alpha.pt')
            torch.save(self.last_alpha,path+ '/last_alpha.pt')
            torch.save(self.best_beta,path+ '/best_beta.pt')
            torch.save(self.last_beta,path+ '/last_beta.pt')

    def save_metrics(self, loss_path, dice_path, alpha_path=None, beta_path=None):
        with open(loss_path, 'wb') as f:
            pickle.dump(self.epoch_loss, f)
        with open(dice_path, 'wb') as f:
            pickle.dump(self.dice_scores, f)
        if alpha_path:
            with open(alpha_path, 'wb') as f:
                pickle.dump(self.alpha_list, f)
        if alpha_path:
            with open(beta_path, 'wb') as f:
                pickle.dump(self.beta_list, f)

def load_metrics(path):
    with open(path, 'rb') as f:
        var = pickle.load(f)
    return var

