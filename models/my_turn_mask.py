import logging
from turtle import end_fill
#from types import NoneType
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.baseline import Baseline
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8
T = 2
lamda = 1
fishermax = 0.0005#0.0001
yigou = 1
k = 4 #4
k1 = 4
t1,t2,t3 = [10,15,20] 
class MyEWC(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._batch_size = args["batch_size"]
        self._num_workers = args["workers"]
        self._lr = args["lr"]
        self._epochs = args["epochs"]
        self._momentum = args["momentum"]
        self._weight_decay = args["weight_decay"]
        self._lr_steps = args["lr_steps"]
        self._modality = args["modality"]

        self._partialbn = args["partialbn"]
        self._freeze = args["freeze"]
        self._clip_gradient = args["clip_gradient"]

        self.fisher = None
        self._network = Baseline(args["num_segments"], args["modality"], args["arch"],
                                          consensus_type=args["consensus_type"],
                                          dropout=args["dropout"], midfusion=args["midfusion"])
        self.fishermax = fishermax
        self.rgb_s,self.flow_s,self.acc_s,self.gyro_s = [0,0,0,0]
    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self.fisher is None:
            self.fisher = self.getFisherDiagonal(self.train_loader)

        else:
            alpha = self._known_classes / self._total_classes
            new_finsher = self.getFisherDiagonal(self.train_loader)
            for n, p in new_finsher.items():
                new_finsher[n][: len(self.fisher[n])] = (
                        alpha * self.fisher[n]
                        + (1 - alpha) * new_finsher[n][: len(self.fisher[n])]
                )
        
            self.fisher = new_finsher

        self.mean = {
            n: p.clone().detach()
            for n, p in self._network.named_parameters()
            if p.requires_grad
        }
     
      
    
    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
            
        optimizer = self._choose_optimizer()

        if type(optimizer) == list:
            scheduler_adam = optim.lr_scheduler.MultiStepLR(optimizer[0], self._lr_steps, gamma=0.1)
            scheduler_sgd = optim.lr_scheduler.MultiStepLR(optimizer[1], self._lr_steps, gamma=0.1)
            scheduler = [scheduler_adam, scheduler_sgd]
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, self._lr_steps, gamma=0.1)

        if self._cur_task == 0:
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            if self._partialbn:
                self._network.feature_extract_network.freeze_fn('partialbn_statistics')
            if self._freeze:
                self._network.feature_extract_network.freeze_fn('bn_statistics')

            losses = 0.0
            correct, total = 0, 0
          
            if epoch % 5 == 0 and epoch >= t1:
                    self.tmp_fisher = self.getFisherDiagonal(self.train_loader)
            for i, (_, inputs, targets) in enumerate(train_loader):
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)
                #logits = self._network(inputs)["logits"]
                comfeat = self._network.feature_extractor(inputs)["features"]
           
                if 5>= epoch >= 0:
                    for name, parms in self._network.named_parameters():
                        if '.flow.' or '.stft' or 'stft_2' in name:
                            parms.requires_grad = False
                        if name == 'fc.weight':
                            parms[:,1024:4096] = 0

                if t3 > epoch > t1:
          
    
                      th_u_r = comfeat[:,:1024].mean() + k * comfeat[:,:1024].std()
                      th_d_r = comfeat[:,:1024].mean() - k * comfeat[:,:1024].std()
                      #print("before:",torch.abs(comfeat[:,:1024]).mean())
                      comfeat[:,:1024][th_d_r > comfeat[:,:1024]] = 0
                      comfeat[:,:1024][comfeat[:,:1024]> th_u_r] = 0
      
            
                      for name, parms in self._network.named_parameters():
                        if '.rgb.' in name:
                            parms.requires_grad = False

                if t3 > epoch > t2:
                    

                      th_u_f = comfeat[:,1025:2048].mean() + k1* comfeat[:,1025:2048].std()
                      th_d_f = comfeat[:,1025:2048].mean() - k1 * comfeat[:,1025:2048].std()

                      comfeat[:,1025:2048][th_d_f > comfeat[:,1025:2048]]= 0
                      comfeat[:,1025:2048][comfeat[:,1025:2048]> th_u_f] = 0
                      comfeat[:,1025:2048] = comfeat[:,1025:2048]
                      for name, parms in self._network.named_parameters():
                        if '.flow.' in name:
                            parms.requires_grad = False
              
                            
                if epoch >= t3:            
                    for name, parms in self._network.named_parameters():
                        if '.flow.' in name or '.rgb.' in name:
                            parms.requires_grad = True
                

                logits = self._network.fc(comfeat)["logits"]
                loss_clf = F.cross_entropy(logits, targets)

                loss = loss_clf

                if type(optimizer) == list:
                    optimizer[0].zero_grad()
                    optimizer[1].zero_grad()
                else:
                    optimizer.zero_grad()

                loss.backward()
                 ##################################################################     
                
                if  epoch >= t1: 
                    m = 0.01
                    rgb_score, flow_score, acc_score, gyro_score = self.init_eval()
                    total_score = rgb_score + flow_score + acc_score + gyro_score
                    r = rgb_score/total_score
                    f = flow_score/total_score
                    a = acc_score/total_score
                    g = gyro_score/total_score
                    #negtive modulation
                    r_ratio= 1/(r + m)
                    f_ratio = 1/(f + m)
                    a_ratio = 1/(a + m) 
                    g_ratio = 1/(g + m)
                    total_ratio = r_ratio + f_ratio + a_ratio + g_ratio
                    r_r = (r_ratio/total_ratio)*4
                    r_f = (f_ratio/total_ratio)*4
                    r_a = (a_ratio/total_ratio)*4
                    r_g = (g_ratio/total_ratio)*4

                    #r_r[r_r>1] *= 5
                    #r_f[r_f>1] *= 5
                    #r_a[r_a>1] *= 5
                    #r_g[r_g>1] *= 5

                   
                    for name, parms in self._network.named_parameters():
                        if '.rgb.' in name and parms.grad is not None:
                            parms.grad = parms.grad * r_r + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-12)
                        elif '.flow.' in name and parms.grad is not None:
                            parms.grad = parms.grad * r_f + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-12)   
                        elif '.stft.' in name and parms.grad is not None:
                            parms.grad = parms.grad * r_a * yigou  + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-12)          
                        elif '.stft_2.' in name and parms.grad is not None:
                            parms.grad = parms.grad * r_g * yigou + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-12)   
                      
                if self._clip_gradient is not None:
                        total_norm = nn.utils.clip_grad_norm_(self._network.parameters(), self._clip_gradient)
               
                
            

                if type(optimizer) == list:
                    optimizer[0].step()
                    optimizer[1].step()
                else:
                    optimizer.step()

                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if type(scheduler) == list:
                scheduler[0].step()
                scheduler[1].step()
            else:
                scheduler.step()
#############################################################################
            #self.tmp_fisher = self.getFisherDiagonal(self.train_loader)
            #rgb_score, flow_score, acc_score, gyro_score = self.init_eval()
            #total_score = rgb_score + flow_score + acc_score + gyro_score
            #rgb_score = rgb_score/total_score
            #flow_score = flow_score/total_score
            #acc_score = acc_score/total_score
            #gyro_score = gyro_score/total_score
            #loss_balance = (0.25 - rgb_score ).pow(2) + (0.25 - flow_score ).pow(2) + (0.25 - acc_score ).pow(2) + (0.25 - gyro_score ).pow(2)
            #loss_balance = total_score
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            if self._partialbn:
                self._network.feature_extract_network.freeze_fn('partialbn_statistics')
            if self._freeze:
                self._network.feature_extract_network.freeze_fn('bn_statistics')

            losses = 0.0
            losses_ewc = 0.0
            correct, total = 0, 0
        
#####################################################################################
            if epoch % 5 == 0 and epoch != 0:
                    self.tmp_fisher = self.getFisherDiagonal(self.train_loader)

            for i, (_, inputs, targets) in enumerate(train_loader):
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)
                #logits = self._network(inputs)["logits"]
                comfeat = self._network.feature_extractor(inputs)["features"]

                if t3 > epoch > t1:
                    #comfeat[:,:1024] = torch.zeros_like(comfeat[:,:1024]).normal_(0, comfeat[:,:1024].std().item() + 1e-12)
    
    
                      th_u_r = comfeat[:,:1024].mean() + k * comfeat[:,:1024].std()
                      th_d_r = comfeat[:,:1024].mean() - k * comfeat[:,:1024].std()

                      comfeat[:,:1024][th_d_r > comfeat[:,:1024]] = 0
                      comfeat[:,:1024][comfeat[:,:1024]> th_u_r] = 0
                      comfeat[:,:1024] = comfeat[:,:1024]
                      for name, parms in self._network.named_parameters():
                        if '.rgb.' in name:
                            parms.requires_grad = False

                if t3 > epoch > t2:
                    

                      th_u_f = comfeat[:,1025:2048].mean() + k1 * comfeat[:,1025:2048].std()
                      th_d_f = comfeat[:,1025:2048].mean() - k1 * comfeat[:,1025:2048].std()

                      comfeat[:,1025:2048][th_d_f > comfeat[:,1025:2048]]= 0
                      comfeat[:,1025:2048][comfeat[:,1025:2048]> th_u_f] = 0
                      comfeat[:,1025:2048] = comfeat[:,1025:2048]

                      for name, parms in self._network.named_parameters():
                        if '.flow.' in name:
                            parms.requires_grad = False
                            
                if epoch >= t3:            
                    for name, parms in self._network.named_parameters():
                        if '.flow.' in name or '.rgb.' in name:
                            parms.requires_grad = True
                          
                logits = self._network.fc(comfeat)["logits"]
                
                #############################################
                #old_features = self._old_network(inputs)["features"]
                #mse_loss = nn.MSELoss(reduction='sum')
                #distill_loss = (mse_loss(self._network(inputs)["features"], self._old_network(inputs)["features"])/100000)*0.1
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes:], targets - self._known_classes
                )

                #rgb_score, flow_score, acc_score, gyro_score, fc_score = self.multimodal_ewc_eval()

                #_rgb = (rgb_score/(rgb_score + flow_score + acc_score + gyro_score))
                #_flow = (flow_score/(rgb_score + flow_score + acc_score + gyro_score))
                #_acc = (acc_score/(rgb_score + flow_score + acc_score + gyro_score))
                #_gyro = (gyro_score/(rgb_score + flow_score + acc_score + gyro_score))

                loss_ewc = self.compute_ewc() * 5e6#50000
                ##################################################
                #loss_ewc_rgb,loss_ewc_flow,loss_ewc_acc,loss_ewc_gyro = self.compute_ewc_mul()
                #loss_ewc = (loss_ewc_rgb + loss_ewc_flow + loss_ewc_acc + loss_ewc_gyro) * 50000
                
                
                #loss_params_eval = self.multimodal_ewc_eval() * 100000
                loss = loss_clf  + lamda * loss_ewc 

                if type(optimizer) == list:
                    optimizer[0].zero_grad()
                    optimizer[1].zero_grad()
                else:
                    optimizer.zero_grad()

                loss.backward()
                
                if epoch > t1: 
                    m = 0.01
                    rgb_score, flow_score, acc_score, gyro_score = self.init_eval()
                    total_score = rgb_score + flow_score + acc_score + gyro_score
                    r = rgb_score/total_score
                    f = flow_score/total_score
                    a = acc_score/total_score
                    g = gyro_score/total_score
                    #negtive modulation
                    r_ratio= 1/(r + m)
                    f_ratio = 1/(f + m)
                    a_ratio = 1/(a + m) 
                    g_ratio = 1/(g + m)
                    total_ratio = r_ratio + f_ratio + a_ratio + g_ratio
                    r_r = (r_ratio/total_ratio)*4
                    r_f = (f_ratio/total_ratio)*4
                    r_a = (a_ratio/total_ratio)*4
                    r_g = (g_ratio/total_ratio)*4
 
                  
              
                    for name, parms in self._network.named_parameters():
                        if '.rgb.' in name and parms.grad is not None:
                            parms.grad = parms.grad * r_r + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-12)
                        elif '.flow.' in name and parms.grad is not None:
                            parms.grad = parms.grad * r_f + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-12)  
                        elif '.stft.' in name and parms.grad is not None:
                            parms.grad= parms.grad * r_a * yigou + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-12)    
                        elif '.stft_2.' in name and parms.grad is not None:
                            parms.grad = parms.grad * r_g * yigou+ torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-12)   
                #else:
                #    pass   
              
                if self._clip_gradient is not None:
                        total_norm = nn.utils.clip_grad_norm_(self._network.parameters(), self._clip_gradient)

                
                if type(optimizer) == list:
                    optimizer[0].step()
                    optimizer[1].step()
                else:
                    optimizer.step()
                losses += loss.item()
                losses_ewc += loss_ewc.item()
 #############################################################               
              
##############################################################plastic
             

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if type(scheduler) == list:
                scheduler[0].step()
                scheduler[1].step()
            else:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, EWC_loss {:.3f},Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs, 
                    losses / len(train_loader),
                    losses_ewc / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, EWC_loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    losses_ewc / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def compute_ewc(self):
        loss = 0
        if len(self._multiple_gpus) > 1:
            for n, p in self._network.named_parameters():
                if n in self.fisher.keys():
                    loss += (
                            torch.sum(
                                (self.fisher[n])
                                * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )
                            / 2
                    )
        else:
            for n, p in self._network.named_parameters():
                if n in self.fisher.keys():
                    loss += (
                            torch.sum(
                                (self.fisher[n])
                                * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )
                            / 2
                    )
                 
        return loss

    def compute_ewc_mul(self):
        loss_rgb = 0
        loss_flow = 0
        loss_acc = 0
        loss_gyro = 0
        if len(self._multiple_gpus) > 1:
            for n, p in self._network.named_parameters():
                if n in self.fisher.keys():
                    loss += (
                            torch.sum(
                                (self.fisher[n])
                                * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )
                            / 2
                    )
        else:
            for n, p in self._network.named_parameters(): #### params eval
                if n in self.fisher.keys() and ".rgb." in n:
                    loss_rgb += (
                            torch.sum(
                                (self.fisher[n])
                                * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )
                            / 2
                    )
                elif n in self.fisher.keys() and ".flow." in n:
                    loss_flow += (
                            torch.sum(
                                (self.fisher[n])
                                * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )
                            / 2
                    )
                elif n in self.fisher.keys() and ".acce." in n:
                    loss_acc += (
                            torch.sum(
                                (self.fisher[n])
                                * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )
                            / 2
                    )    
                elif n in self.fisher.keys() and ".gyro." in n:
                    loss_gyro += (
                            torch.sum(
                                (self.fisher[n])
                                * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )
                            / 2
                    )   
                  
        return {loss_rgb, loss_flow, loss_acc, loss_gyro}

    def multimodal_ewc_eval(self):
        loss_rgb = []
        loss_flow = []
        loss_acc = []
        loss_gyro = []
        if len(self._multiple_gpus) > 1:
            for n, p in self._network.named_parameters():
                if n in self.tmp_fisher.keys():
                    loss += (
                            torch.sum(
                                (self.tmp_fisher[n])
                            )
                            
                    )
        else:
            for n, p in self._network.named_parameters(): #### params eval
                if n in self.tmp_fisher.keys() and ".rgb." in n:
                    loss_rgb.append (
                            torch.mean(
                                (self.tmp_fisher[n])
                            )  
                             
                    )


                elif n in self.tmp_fisher.keys() and ".flow." in n:
                    loss_flow.append (
                            torch.mean(
                                (self.tmp_fisher[n])
                            )    
                            
                    )
               
                elif n in self.tmp_fisher.keys() and ".acce." in n:
                    loss_acc.append (
                            torch.mean(
                                (self.tmp_fisher[n])
                            )    
                                    
                    )    
                
                elif n in self.tmp_fisher.keys() and ".gyro." in n:
                    loss_gyro.append (
                            torch.mean(
                                (self.tmp_fisher[n])

                            )    
                            
                    )   

            rgb_avg = torch.mean(loss_rgb)
            flow_avg = torch.mean(loss_flow)
            acc_avg = torch.mean(loss_acc)
            gyro_avg = torch.mean(loss_gyro)
        return {rgb_avg*100, flow_avg*100, acc_avg*100, gyro_avg*100}

    def getFisherDiagonal(self, train_loader):
        for n, p in self._network.named_parameters():
          if '.rgb.' or '.flow.' or '.STFT'or 'STFT_2' in n:
             p.requires_grad = True
        fisher = {
            n: torch.zeros(p.shape).to(self._device)
            for n, p in self._network.named_parameters()
            if p.requires_grad
        }
        
        self._network.train()

        #optimizer = optim.SGD(self._network.parameters(), lr=self._lr)
        optimizer = self._choose_optimizer()
        for i, (_, inputs, targets) in enumerate(train_loader):
            for m in self._modality:
                inputs[m] = inputs[m].to(self._device)
            targets = targets.to(self._device)

            if self._partialbn:
               self._network.feature_extract_network.freeze_fn('partialbn_statistics')
            if self._freeze:
               self._network.feature_extract_network.freeze_fn('bn_statistics')
            #comfeat = self._network.feature_extractor(inputs)["features"]
            #comfeat = comfeat[:,:1024] + torch.zeros_like(comfeat[:,:1024]).normal_(0, comfeat[:,:1024].std().item() + 1e-12)
            #logits = self._network.fc(comfeat)["logits"]
            logits = self._network(inputs)["logits"]
            #if self._cur_task != 0:
            #   loss_ewc = self.compute_ewc()
            #else:
            #   loss_ewc = 0
            loss = torch.nn.functional.cross_entropy(logits, targets)#+ loss_ewc * 50000

            #optimizer.zero_grad()
            #############################################################
            if type(optimizer) == list:
                    optimizer[0].zero_grad()
                    optimizer[1].zero_grad()
            else:
                    optimizer.zero_grad()

            loss.backward()

            for n, p in self._network.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
        for n, p in fisher.items():
            fisher[n] = p / len(train_loader)
            fisher[n] = torch.min(fisher[n], torch.tensor(fishermax))
        return fisher

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            for m in self._modality:
                inputs[m] = inputs[m].to(self._device)
            with torch.no_grad():
                comfeat = model.feature_extractor(inputs)["features"]
                outputs = model.fc(comfeat)["logits"]
                #outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            for m in self._modality:
                inputs[m] = inputs[m].to(self._device)
            with torch.no_grad():
                comfeat = self._network.feature_extractor(inputs)["features"]
                outputs = self._network.fc(comfeat)["logits"]
                #outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def init_eval(self):
        loss_rgb = 0
        loss_flow = 0
        loss_acc = 0
        loss_gyro = 0
        if len(self._multiple_gpus) > 1:
            for n, p in self._network.named_parameters():
                if n in self.tmp_fisher.keys():
                    loss += (
                            torch.mean(
                                (self.tmp_fisher[n])
                                #* (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )
                            
                    )
        else:
            for n, p in self._network.named_parameters(): #### params eval
                if n in self.tmp_fisher.keys() and ".rgb." in n:
                    loss_rgb += (
                            torch.mean(
                                (self.tmp_fisher[n])
                              #  * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )  
                             
                    )
                      
                elif n in self.tmp_fisher.keys() and ".flow." in n:
                    loss_flow += (
                            torch.mean(
                                (self.tmp_fisher[n])
                               # * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )    
                            
                    )
                    
                elif n in self.tmp_fisher.keys() and ".stft." in n:
                    loss_acc += (
                            torch.mean(
                                (self.tmp_fisher[n])
                                #* (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )    
                                    
                    )    
             
                elif n in self.tmp_fisher.keys() and ".stft_2." in n:
                    loss_gyro += (
                            torch.mean(
                                (self.tmp_fisher[n])
                               # * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )    
                            
                    )   


        return loss_rgb*100, loss_flow*100, loss_acc*100, loss_gyro*100
