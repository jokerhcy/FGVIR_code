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
from utils.mydata_manager import MyDataManager
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from torchvision.utils import save_image
#from sklearn.metrics import ConfusionMatrixDisplay
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip,ColorJitter
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
from utils.mydata_manager import MyDataManager
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from torchvision.utils import save_image
#from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.manifold import TSNE
# For the UCI ML handwritten digits dataset
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

EPSILON = 1e-8
T = 2
lamda = 1
fishermax = 0.0005#0.0001
yigou = 1
k = 4 #4
k1 = 4

class MyEWC(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._protos = []
        self._std = []
        self._radius = 0
        self._radiuses = []
        self.rgb_radiuses = []
        self.flow_radiuses = []
        self.acc_radiuses = []
        self.gyro_radiuses = []
        self._batch_size = args["batch_size"]
        self._num_workers = args["workers"]
        self._lr = args["lr"]
        self._epochs = args["epochs"]
        self._momentum = args["momentum"]
        self._weight_decay = args["weight_decay"]
        self._lr_steps = args["lr_steps"]
        self._modality = args["modality"]
        self._num_segments = args["num_segments"]
        self._partialbn = args["partialbn"]
        self._freeze = args["freeze"]
        self._clip_gradient = args["clip_gradient"]

        self.fisher = None
        self._network = Baseline(args["num_segments"], args["modality"], args["arch"],
                                          consensus_type=args["consensus_type"],
                                          dropout=args["dropout"], midfusion=args["midfusion"])
        self.fishermax = fishermax
        self.rgb_s,self.flow_s,self.acc_s,self.gyro_s = [0,0,0,0]
        self.transform = self.transform = nn.Sequential(
            RandomResizedCrop(size=(224, 224), scale=(0.8, 1.)),
            ColorJitter(),
            RandomHorizontalFlip(),
        )

        self.flow_transform = self.transform = nn.Sequential(
            RandomResizedCrop(size=(224, 224), scale=(0.8, 1.)),
            RandomHorizontalFlip(),
        )

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        

    def incremental_train(self, data_manager,args):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes, self._known_classes)
        self._network_module_ptr = self._network
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
        self._train(self.train_loader, self.test_loader,args)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        """ if self.fisher is None:
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
        }"""
     
      
    
    def _train(self, train_loader, test_loader,args):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        optimizer = self._choose_optimizer()

        if type(optimizer) == list:
            scheduler_adam = optim.lr_scheduler.MultiStepLR(optimizer[0], self._lr_steps, gamma=0.1)
            #scheduler_sgd = optim.lr_scheduler.MultiStepLR(optimizer[1], self._lr_steps, gamma=0.1)
            scheduler_sgd = optim.lr_scheduler.CosineAnnealingLR(
                optimizer[1], 10)
            scheduler = [scheduler_adam, scheduler_sgd]
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, self._lr_steps, gamma=0.1)

        if self._cur_task == 0:
            self._init_train(train_loader, test_loader, optimizer, scheduler,args)
        else:
            self._update_representation(train_loader, test_loader, optimizer, scheduler,args)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler,args):
        prog_bar = tqdm(range(15))
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            if self._partialbn:
                self._network.feature_extract_network.freeze_fn('partialbn_statistics')
            if self._freeze:
                self._network.feature_extract_network.freeze_fn('bn_statistics')

            losses = 0.0
            correct, total = 0, 0
          
            #if epoch % 5 == 0 and epoch >= 5:
                    #self.tmp_fisher = self.getFisherDiagonal(self.train_loader)
            for i, (_, inputs, targets) in enumerate(train_loader):
                for m in self._modality:
                    if m =='RGB':
                        #print(inputs[m].size())
                        #save_image(inputs[m][:,:3,:,:],"hcytst.png")
                        #save_image(inputs[m][:,3:6,:,:],"hcytst1.png")
                        #save_image(inputs[m][:,6:9,:,:],"hcytst2.png")
                        #save_image(inputs[m][:,9:12,:,:],"hcytst3.png")
                        frame_1 = self.transform(inputs[m][:,:3,:,:])                       
                        frame_2 = self.transform(inputs[m][:,3:6,:,:])
                        frame_3 = self.transform(inputs[m][:,6:9,:,:])
                        frame_4 = self.transform(inputs[m][:,9:12,:,:])
                        frame_5 = self.transform(inputs[m][:,12:15,:,:])
                        frame_6 = self.transform(inputs[m][:,15:18,:,:])
                        frame_7 = self.transform(inputs[m][:,18:21,:,:])
                        frame_8 = self.transform(inputs[m][:,21:24,:,:])
                        trans_inputs = torch.cat([frame_1,frame_2,frame_3,frame_4,frame_5,frame_6,frame_7,frame_8],dim=1)
                        inputs[m] = torch.cat([inputs[m], trans_inputs]).to(self._device)
                        #print(inputs[m].size())
                    elif m =='Flow':
                        inputs[m] = torch.cat([inputs[m], self.flow_transform(inputs[m])]).to(self._device)
                    else:
                        inputs[m] = torch.cat([inputs[m], inputs[m]]).to(self._device)
                
                #logits = self._network(inputs)["logits"]
                feat = self._network.feature_extractor(inputs)["features"]
                time_batch = feat.size()[0]
               
                f1, f2 = torch.split(feat, [int(time_batch/2), int(time_batch/2)], dim=0)

                con_loss = SupCon(self, f1, f2, targets, 0.07)
                targets = targets.to(self._device)
                
                
                comfeat = f1.clone()
                #seperate train:
                """if 5 >= epoch >= 0:
                    for name, parms in self._network.named_parameters():
                      if '.flow.' in name or '.stft.' in name or '.stft_2.' in name:
                            parms.requires_grad = False
                    
                    logits = self._network(inputs)["RGB_logits"]

                if 10 >= epoch > 5:
                    for name, parms in self._network.named_parameters():
                      if '.rgb.' in name or '.stft.' in name or '.stft_2.' in name:
                            parms.requires_grad = False
                    
                    logits = self._network(inputs)["flow_logits"]
                
                if 15 >= epoch > 10:
                    for name, parms in self._network.named_parameters():
                      if '.flow.' in name or '.rgb.' in name or '.stft_2.' in name:
                            parms.requires_grad = False
                    
                    logits = self._network(inputs)["acc_logits"]

                if 20 >= epoch > 15:
                    for name, parms in self._network.named_parameters():
                      if '.flow.' in name or '.stft.' in name or '.rgb.' in name:
                            parms.requires_grad = False
                    
                    logits = self._network(inputs)["gyro_logits"]
                
                if 30 >= epoch > 20:
                    for name, parms in self._network.named_parameters():
                      if '.flow.' in name or '.rgb.'in name or '.stft.' in name or '.stft_2.' in name:
                            parms.requires_grad = False

                    logits = self._network(inputs)["logits"]

                else:
                    logits = self._network(inputs)["logits"]"""
                if 15 > epoch > 5:
                    #comfeat[:,:1024] = torch.zeros_like(comfeat[:,:1024]).normal_(0, comfeat[:,:1024].std().item() + 1e-12)
    
                      th_u_r = comfeat[:,:1024].mean() + k * comfeat[:,:1024].std()
                      th_d_r = comfeat[:,:1024].mean() - k * comfeat[:,:1024].std()
                      #print("before:",torch.abs(comfeat[:,:1024]).mean())
                      comfeat[:,:1024][th_d_r > comfeat[:,:1024]] = 0
                      comfeat[:,:1024][comfeat[:,:1024]> th_u_r] = 0
                      #print("aftermask:",torch.abs(comfeat[:,:1024].mean()))
                      comfeat[:,:1024] = comfeat[:,:1024]
                      for name, parms in self._network.named_parameters():
                        if '.rgb.' in name:
                            parms.requires_grad = False

                if 15 > epoch > 10:
                    

                      th_u_f = comfeat[:,1024:2048].mean() + k1* comfeat[:,1024:2048].std()
                      th_d_f = comfeat[:,1024:2048].mean() - k1 * comfeat[:,1024:2048].std()

                      comfeat[:,1024:2048][th_d_f > comfeat[:,1024:2048]]= 0
                      comfeat[:,1024:2048][comfeat[:,1024:2048]> th_u_f] = 0
                      comfeat[:,1024:2048] = comfeat[:,1024:2048]
                      for name, parms in self._network.named_parameters():
                        if '.flow.' in name:
                            parms.requires_grad = False

                    #comfeat[:,1025:2048] + 0.1*torch.zeros_like(comfeat[:,1025:2048]).normal_(0, comfeat[:,1025:2048].std().item() + 1e-12)
                    #for name, parms in self._network.named_parameters():
                    #    if '.rgb.' in name:
                    #        parms.requires_grad = False
                            
                if epoch > 15:            
                    for name, parms in self._network.named_parameters():
                        if '.flow.' in name or '.rgb.' in name:
                            parms.requires_grad = True

                logits = self._network.fc(comfeat)["logits"]
                
                
                loss_clf = F.cross_entropy(logits, targets)

                if 0 <= epoch < 10:
                    loss = loss_clf + 0.1*con_loss
                else:
                    loss = loss_clf
                    
                if type(optimizer) == list:
                    optimizer[0].zero_grad()
                    optimizer[1].zero_grad()
                else:
                    optimizer.zero_grad()

                loss.backward()
                 ##################################################################     
                """
                if  epoch > 5: 
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
                            parms.grad = parms.grad * r_a*yigou  + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-12)          
                        elif '.stft_2.' in name and parms.grad is not None:
                            parms.grad = parms.grad * r_g*yigou  + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-12)   """

                if self._clip_gradient is not None:
                        total_norm = nn.utils.clip_grad_norm_(self._network.parameters(), self._clip_gradient)
                #else:
                #    pass   

            

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
                    30,
                    losses / len(train_loader),
                    train_acc,
                )
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    30,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)
        image_tmpl = {}
        for m in args["modality"]:
        # Prepare dictionaries containing image name templates for each modality
          if m in ['RGB', 'RGBDiff']:
            image_tmpl[m] = "img_{:06d}.jpg"
          elif m == 'Flow':
            image_tmpl[m] = args["flow_prefix"] + "{}_{:06d}.jpg"
          elif m == 'STFT' or m =='STFT_2':
            image_tmpl[m] = "{}_{:01d}.jpg"
        
        data_manager = MyDataManager(self, image_tmpl, args)
        self.protoSave(self._network, self.train_loader, data_manager)
        
    def _update_representation(self, train_loader, test_loader, optimizer, scheduler,args):
        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            if self._partialbn:
                self._network.feature_extract_network.freeze_fn('partialbn_statistics')
            if self._freeze:
                self._network.feature_extract_network.freeze_fn('bn_statistics')

            losses = 0.0
            losses_ewc = 0.0
            losses_clf = 0.0
            correct, total = 0, 0
            losses_protoAug = 0.0
            losses_dis =0.0
#####################################################################################
            
            #if epoch % 5 == 0 and epoch != 0:
            #        self.tmp_fisher = self.getFisherDiagonal(self.train_loader)

            for i, (_, inputs, targets) in enumerate(train_loader):
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)
                #logits = self._network(inputs)["logits"]
                #comfeat = self._network.feature_extractor(inputs)["features"]
                features = self._network.extract_vector(inputs)
                features_old = self._old_network.extract_vector(inputs)
                loss_dis = torch.dist(features, features_old, 2)*0
                
                """
                if 15 > epoch > 5:
                    #comfeat[:,:1024] = torch.zeros_like(comfeat[:,:1024]).normal_(0, comfeat[:,:1024].std().item() + 1e-12)
                    for dim in range(1024):
    
                      th_u_r = comfeat[:,:1024].mean() + k * comfeat[:,:1024].std()
                      th_d_r = comfeat[:,:1024].mean() - k * comfeat[:,:1024].std()

                      comfeat[:,:1024][th_d_r > comfeat[:,:1024]] = 0
                      comfeat[:,:1024][comfeat[:,:1024]> th_u_r] = 0
                      comfeat[:,:1024] = comfeat[:,:1024]
                    for name, parms in self._network.named_parameters():
                        if '.rgb.' in name:
                            parms.requires_grad = False

                if 15 > epoch > 10:
                    for dim in range(1024):

                      th_u_f = comfeat[:,1025:2048].mean() + k1 * comfeat[:,1025:2048].std()
                      th_d_f = comfeat[:,1025:2048].mean() - k1 * comfeat[:,1025:2048].std()

                      comfeat[:,1025:2048][th_d_f > comfeat[:,1025:2048]]= 0
                      comfeat[:,1025:2048][comfeat[:,1025:2048]> th_u_f] = 0
                      comfeat[:,1025:2048] = comfeat[:,1025:2048]

                    for name, parms in self._network.named_parameters():
                        if '.flow.' in name:
                            parms.requires_grad = False
                            
                if epoch >= 15:            
                    for name, parms in self._network.named_parameters():
                        if '.flow.' in name or '.rgb.' in name:
                            parms.requires_grad = True
                            
                """
                #logits = self._network.fc(comfeat)["logits"]
                logits = self._network(inputs)["logits"]
                #############################################
                #old_features = self._old_network(inputs)["features"]
                #mse_loss = nn.MSELoss(reduction='sum')
                #distill_loss = (mse_loss(self._network(inputs)["features"], self._old_network(inputs)["features"])/100000)*0.1
                #a = logits[:, self._known_classes:]
                #f = targets - self._known_classes
                for name, parms in self._network.named_parameters():
                      if '.flow.' in name or '.rgb.'in name or '.stft.' in name or '.stft_2.' in name or '.acce.' in name or '.gyro.' in name:
                            parms.requires_grad = False

                #loss_clf = torch.nn.functional.cross_entropy(logits, targets)
                #loss_clf = F.cross_entropy(
                #   logits[:, self._known_classes:], targets - self._known_classes
                #)

                loss_clf = F.cross_entropy(logits, targets)
             
                #+ F.cross_entropy(
                 
                #  acc_logits[:, self._known_classes :], fake_targets
                #)

##################################################################################
                loss_kd = _KD_loss(logits[:, : self._known_classes], self._old_network(inputs)["logits"],T,)*0
                #rgb_score, flow_score, acc_score, gyro_score, fc_score = self.multimodal_ewc_eval()

                #_rgb = (rgb_score/(rgb_score + flow_score + acc_score + gyro_score))
                #_flow = (flow_score/(rgb_score + flow_score + acc_score + gyro_score))
                #_acc = (acc_score/(rgb_score + flow_score + acc_score + gyro_score))
                #_gyro = (gyro_score/(rgb_score + flow_score + acc_score + gyro_score))

                #loss_ewc = self.compute_ewc() * 50000#50000
                ##################################################
                #loss_ewc_rgb,loss_ewc_flow,loss_ewc_acc,loss_ewc_gyro = self.compute_ewc_mul()
                #loss_ewc = (loss_ewc_rgb + loss_ewc_flow + loss_ewc_acc + loss_ewc_gyro) * 50000
                
               
       #################proto_loss#######################
              
                
                index = np.random.choice(range(self._known_classes),size= self._batch_size * int(self._known_classes/(self._total_classes-self._known_classes)),replace=True)
                proto_features = np.array(self._protos)[index]
                std_features = np.array(self._std)[index]
                proto_targets = index
                proto_time = np.zeros([proto_features.shape[0],8,proto_features.shape[1]])
                for i in range(proto_features.shape[0]):
                    for j in range(8):
                          if len(self._modality) > 1:   
                            proto_time[i,j,:] = proto_features[i,:] + np.random.normal(0,1,proto_features[i,:].shape) * std_features[i,:]*2
                          else:
                            proto_time[i,j,:] = proto_features[i] + np.random.normal(0,1,proto_features[i].shape)*self._radius*2
                proto_time = torch.from_numpy(proto_time).float().to(self._device,non_blocking=True)
                proto_time = proto_time.view([-1,proto_features.shape[1]])
                

                #proto_features = proto_features + np.random.normal(0,1,proto_features.shape)*self._radius*2
                #proto_aug = torch.from_numpy(proto_features).float().to(self._device,non_blocking=True)
                proto_targets = torch.from_numpy(proto_targets).to(self._device,non_blocking=True)

                """index = list(range(self._known_classes))
                for _ in range(8):
                    np.random.shuffle(index)
                    temp = self.prototype[index[0]] + np.random.normal(0, 1, 4096) * self.radius
                    proto_aug.append(temp)
                    proto_aug_label.append(self.class_label[index[0]])"""
                    #proto_aug_label.append(self.class_label[index[0]])

                #proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self._device)
                #proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self._device)
                #proto_aug_group =  torch.cat([proto_aug,proto_aug,proto_aug,proto_aug,proto_aug,proto_aug,proto_aug,proto_aug],dim=0)
                
                """proto_chart_0 = proto_aug[0].repeat(8,1)
                proto_chart_1 = proto_aug[1].repeat(8,1)
                proto_chart_2 = proto_aug[2].repeat(8,1)
                proto_chart_3 = proto_aug[3].repeat(8,1)
                proto_chart_4 = proto_aug[4].repeat(8,1)
                proto_chart_5 = proto_aug[5].repeat(8,1)
                proto_chart_6 = proto_aug[6].repeat(8,1)
                proto_chart_7 = proto_aug[7].repeat(8,1)
                proto_aug_group =  torch.cat([ proto_chart_0,proto_chart_1,proto_chart_2,proto_chart_3,proto_chart_4,
                                                proto_chart_5,proto_chart_6,proto_chart_7],dim=0)"""


                soft_feat_aug = self._network.fc(proto_time)["logits"]
                #soft_feat_aug = self._network.fc(proto_aug_group)["logits"]
                loss_protoAug =  F.cross_entropy(soft_feat_aug, proto_targets)*1e5
                #loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug/self.args.temp, proto_aug_label)
                #loss_protoAug = 0.05 * nn.CrossEntropyLoss()(soft_feat_aug, proto_aug_label - self._known_classes)
                #loss_protoAug =  F.cross_entropy(soft_feat_aug, proto_aug_label)
                #loss_params_eval = self.multimodal_ewc_eval() * 100000
                #loss = loss_clf  + lamda * loss_ewc
                loss = loss_clf  + loss_protoAug + loss_dis + loss_kd
                
                if type(optimizer) == list:
                    optimizer[0].zero_grad()
                    optimizer[1].zero_grad()
                else:
                    optimizer.zero_grad()

                loss.backward()
                """
                if epoch > 5: 
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
                    #positive modulation
                    #r_ratio= r
                    #f_ratio = f
                    #a_ratio = a
                    #g_ratio = g
                    #total_ratio = r_ratio + f_ratio + a_ratio + g_ratio
                    #r_r = (r_ratio/total_ratio)*4
                    #r_f = (f_ratio/total_ratio)*4
                    #r_a = (a_ratio/total_ratio)*4
                    #r_g = (g_ratio/total_ratio)*4

                #if 0 <= epoch <= 10:
                  

                    for name, parms in self._network.named_parameters():
                        if '.rgb.' in name and parms.grad is not None:
                            parms.grad = parms.grad * r_r + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-12)
                        elif '.flow.' in name and parms.grad is not None:
                            parms.grad = parms.grad * r_f + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-12)  
                        elif '.acce.' in name and parms.grad is not None:
                            parms.grad= parms.grad * r_a *yigou + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-12)    
                        elif '.gyro.' in name and parms.grad is not None:
                            parms.grad = parms.grad * r_g *yigou+ torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-12)   
                #else:
                #    pass   
                """
                if self._clip_gradient is not None:
                        total_norm = nn.utils.clip_grad_norm_(self._network.parameters(), self._clip_gradient)

                
                if type(optimizer) == list:
                    optimizer[0].step()
                    optimizer[1].step()
                else:
                    optimizer.step()
                losses += loss.item()
                #losses_ewc += loss_ewc.item()
                losses_protoAug += loss_protoAug.item()
                losses_dis += loss_dis.item()
                losses_clf +=loss_clf.item()
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
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs, 
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f},  ClfLoss {:.3f} ,PR_loss {:.3f}, KD {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    #losses_ewc / len(train_loader),
                    losses_protoAug / len(train_loader),
                    losses_dis/ len(train_loader),
                    train_acc,
                    test_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

        image_tmpl = {}
        for m in args["modality"]:
        # Prepare dictionaries containing image name templates for each modality
          if m in ['RGB', 'RGBDiff']:
            image_tmpl[m] = "img_{:06d}.jpg"
          elif m == 'Flow':
            image_tmpl[m] = args["flow_prefix"] + "{}_{:06d}.jpg"
          elif m == 'STFT' or m =='STFT_2':
            image_tmpl[m] = "{}_{:01d}.jpg"
        
        data_manager = MyDataManager(self, image_tmpl, args)
        self.protoSave(self._network, self.train_loader, data_manager)
       

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        targets_all =[]
        predicts_all =[]
        for i, (_, inputs, targets) in enumerate(loader):
            for m in self._modality:
                inputs[m] = inputs[m].to(self._device)
            with torch.no_grad():
                #outputs = model(inputs)["logits"]
                comfeat = model.feature_extractor(inputs)["features"]
                outputs = model.fc(comfeat)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            target1 = targets.numpy()
            targets_all.append(target1)
            predicts1 = predicts.cpu().numpy()
            predicts_all.append(predicts1)
            
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        #targets_all = np.hstack(targets_all)
        #predicts_all = np.hstack(predicts_all)
        """targets_all_1 = targets_all.numpy()
        predicts_all_1 = targets_all.cpu().numpy()"""
        #cf = confusion_matrix(targets_all, predicts_all).astype(float)
        #np.savetxt('cfm.txt',cf)
        #if self._cur_task == 1:
        #    types = ['0','1','2','3','4','5','6','7']
        #    plot_confusion_matrix(cf,types,normalize=True)
        #cf_map = ConfusionMatrixDisplay(cf)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            for m in self._modality:
                inputs[m] = inputs[m].to(self._device)
            with torch.no_grad():
                #outputs = self._network(inputs)["logits"]
                comfeat = self._network.feature_extractor(inputs)["features"]
                outputs = self._network.fc(comfeat)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

        
    def protoSave(self, model, train_loader, data_manager):

        rgb_mean = 0
        flow_mean =0
        acc_mean = 0
        gyro_mean = 0
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
              data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
                )
              idx_loader = DataLoader(
                    idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
                )
              if len(self._modality) > 1:   
                vectors, _ = self._extract_vectors(idx_loader)

                class_mean = np.mean(vectors, axis=0)
                class_std = np.std(vectors, axis=0)
                self._protos.append(class_mean)
                self._std.append(class_std)

                rgb_mean += class_mean[:1024].mean()
                flow_mean += class_mean[1024:2048].mean()
                acc_mean += class_mean[2048:3072].mean()
                gyro_mean += class_mean[3072:4096].mean()
                
              else:
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                class_std = np.std(vectors, axis=0)
                self._protos.append(class_mean)
                self._std.append(class_std)

            
    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        
        #RGB_vectors, Acce_vectors, Gyro_vectors = [], [], []
        for _, _inputs, _targets in loader:
            proto_w = []
            for m in self._modality:
                _inputs[m] = _inputs[m].to(self._device)
            _targets = _targets.numpy()
            logits = self._network(_inputs)['logits_pre'].to(self._device)

            if isinstance(self._network, nn.DataParallel):
                for i in range(_targets.shape[0]):
                    
                    ########1.生成[1,-1,-1,...,-1]
                    """one_hot = torch.nn.functional.one_hot(_targets[i], logits.shape[0]) 
                    encode = one_hot + one_hot - torch.ones(1,logits.shape[0])
                    proto_w.append(torch.matmul(encode,logits[i]))"""
                    ########2.生成[1,-1/(n-1),...,-1/(n-1)]
                    proto_w.append(torch.matmul(encode,logits[i]))
                proto_w = proto_w / proto_w.sum()
                _vectors = tensor2numpy(
                    np.dot(proto_w, self._network.module.extract_vector(_inputs))
                )

            else:        
                  
                """ ori_vectors = tensor2numpy(
                    self._consensus(self._network.extract_vector(_inputs))
                )   """
                for i in range(_targets.shape[0]): 
                    ###########1.生成[1,-1,-1,...,-1]############                   
                    one_hot = torch.nn.functional.one_hot(torch.tensor(_targets[i]), logits.shape[2]).to(self._device) 
                    encode = one_hot + one_hot - torch.ones(1,logits.shape[2]).to(self._device)
                    ###########2.生成[1,-1/(n-1),...,-1/(n-1)]###
                    """one_hot = torch.nn.functional.one_hot(torch.tensor(_targets[i]), logits.shape[2]).to(self._device) 
                    encode = one_hot + one_hot - torch.ones(1,logits.shape[2]).to(self._device)
                    one = torch.ones((1,logits.shape[2]))
                    dividend = -(logits.shape[2])+1
                    n = (one/dividend).to(self._device)
                    encode = torch.where(encode == -1,n,encode)"""
                    
                    #encode = encode.repeat(8,1)
                    #print(logits[0].size())
                    #a = torch.matmul(encode,logits[i].t()).t()
                    a = torch.matmul(logits[i],encode.t())
                    #a[a<0] = 0
                    b = a.cpu().numpy()
                    b = b / np.sum(b)
                    proto_w.append(b)
                
                proto_w = np.squeeze(proto_w,axis=2)
                
                c = self._network.extract_vector(_inputs)
                d = c.view((-1,self._num_segments) + c.size()[1:])
                proto_w = np.expand_dims(proto_w,2).repeat(d.shape[2],axis=2)

                _vectors = np.multiply(proto_w, d.cpu().numpy())
                _vectors = np.sum(_vectors,axis=1)
            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    
    def _consensus(self, x):
        output = x.view((-1, self._num_segments) + x.size()[1:])
        output = output.mean(dim=1, keepdim=True)
        output = output.squeeze(1)
        return output
    
def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]



import itertools
import matplotlib.pyplot as plt
import numpy as np
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig('_111.png')
    fig.clear()

#def evalution_proto():
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        #print(batch_size)
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            #print(labels.shape[0])
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def SupCon(self, z1, z2, target, t=0.07):
        z = F.normalize(torch.stack([z1, z2], dim=1), dim=-1)
        return SupConLoss(temperature=t, base_temperature=t)(z)