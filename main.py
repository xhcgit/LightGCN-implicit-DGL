# coding=UTF-8
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ToolScripts.TimeLogger import log
import pickle
import os
import sys
import gc
import random
import argparse
import scipy.sparse as sp
from ToolScripts.utils import mkdir
from ToolScripts.utils import loadData
from dgl import DGLGraph
from LightGCN import MODEL
from BPRData import BPRData
import torch.utils.data as dataloader
import evaluate
import time
import networkx as nx
import dgl


device_gpu = t.device("cuda")
modelUTCStr = str(int(time.time()))[4:]

isLoadModel = False
LOAD_MODEL_PATH = ""

class Model():

    def __init__(self, args, isLoad=False):
        self.args = args
        self.datasetDir = os.path.join(os.path.dirname(os.getcwd()), "dataset", args.dataset, 'implicit', "cv{0}".format(args.cv))

        trainMat, uuMat, iiMat = self.getData(args)
        self.userNum, self.itemNum = trainMat.shape
        log("user num =%d, item num =%d"%(self.userNum, self.itemNum))
        u_i_adj = (trainMat != 0) * 1 
        i_u_adj = u_i_adj.T

        a = sp.csr_matrix((self.userNum, self.userNum))
        b = sp.csr_matrix((self.itemNum, self.itemNum))
        if args.trust == 1:
            adj = sp.vstack([sp.hstack([uuMat, u_i_adj]), sp.hstack([i_u_adj, b])]).tocsr()
        else:
            adj = sp.vstack([sp.hstack([a, u_i_adj]), sp.hstack([i_u_adj, b])]).tocsr()

        log("uu num = %d"%(uuMat.nnz))
        log("ii num = %d"%(iiMat.nnz))
        self.trainMat = trainMat

        edge_src, edge_dst = adj.nonzero()
        
        self.uv_g = dgl.graph(data=(edge_src, edge_dst),
                              idtype=t.int32,
                              num_nodes=adj.shape[0],
                              device=device_gpu)


        #train data
        train_u, train_v = self.trainMat.nonzero()
        assert np.sum(self.trainMat.data ==0) == 0
        log("train data size = %d"%(train_u.size))
        train_data = np.hstack((train_u.reshape(-1,1), train_v.reshape(-1,1))).tolist()
        train_dataset = BPRData(train_data, self.itemNum, self.trainMat, self.args.num_ng, True)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True, num_workers=0)
        
        #test_data
        with open(self.datasetDir + "/test_data.pkl", 'rb') as fs:
            test_data = pickle.load(fs)
        test_dataset = BPRData(test_data, self.itemNum, self.trainMat, 0, False)
        self.test_loader  = dataloader.DataLoader(test_dataset, batch_size=args.test_batch*101, shuffle=False, num_workers=0)
        #valid data
        with open(self.datasetDir + "/valid_data.pkl", 'rb') as fs:
            valid_data = pickle.load(fs)
        valid_dataset = BPRData(valid_data, self.itemNum, self.trainMat, 0, False)
        self.valid_loader  = dataloader.DataLoader(valid_dataset, batch_size=args.test_batch*101, shuffle=False, num_workers=0)

        self.lr = self.args.lr #0.001
        self.curEpoch = 0
        self.isLoadModel = isLoad
        #history
        self.train_loss = []
        self.his_hr = []
        self.his_ndcg  = []
        gc.collect()
        log("gc.collect()")

    def setRandomSeed(self):
        np.random.seed(self.args.seed)
        t.manual_seed(self.args.seed)
        t.cuda.manual_seed(self.args.seed)
        random.seed(self.args.seed)
    
    def getData(self, args):
        trainMat = loadData(args.dataset, args.cv)

        with open(self.datasetDir + '/uu_vv_graph.pkl', 'rb') as fs:
            uu_vv_graph = pickle.load(fs)
        uuMat = uu_vv_graph['UU'].astype(np.bool)
        iiMat = uu_vv_graph['II'].astype(np.bool)
        return trainMat, uuMat, iiMat

    #初始化参数
    def prepareModel(self):
        self.modelName = self.getModelName() 
        self.setRandomSeed()

        # self.layer = eval(self.args.layer)
        self.hide_dim = args.hide_dim
        self.out_dim = self.hide_dim
        
        self.model = MODEL(self.args, self.userNum, self.itemNum, self.hide_dim, self.args.layerNum).cuda()


        self.opt = t.optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay=0)

    def adjust_learning_rate(self, opt, epoch):
        for param_group in opt.param_groups:
            param_group['lr'] = max(param_group['lr'] * self.args.decay, self.args.minlr)
            # log("cur lr = %.6f"%(param_group['lr']))
    
    def innerProduct(self, u, i, j):
        pred_i = t.sum(t.mul(u,i), dim=1)
        pred_j = t.sum(t.mul(u,j), dim=1)
        return pred_i, pred_j
    
    def run(self):
        #判断是导入模型还是重新训练模型
        self.prepareModel()
        if self.isLoadModel == True:
            self.loadModel(LOAD_MODEL_PATH)
            # HR, NDCG = self.validModel(self.test_loader,save=False)
            HR, NDCG = self.validModel(self.valid_loader,save=False)
            # with open(self.datasetDir + "/test_data.pkl".format(self.args.cv), 'rb') as fs:
            #     test_data = pickle.load(fs)
            # uids = np.array(test_data[::101])[:,0]
            # data = {}
            # assert len(uids) == len(HR)
            # assert len(uids) == len(np.unique(uids))
            # for i in range(len(uids)):
            #     uid = uids[i]
            #     data[uid] = [HR[i], NDCG[i]]

            # with open("KCGN-{0}-cv{1}-test.pkl".format(self.args.dataset, self.args.cv), 'wb') as fs:
            #     pickle.dump(data, fs)

            log("HR = %.4f, NDCG = %.4f"%(np.mean(HR), np.mean(NDCG)))
            # return
        cvWait = 0
        best_HR = 0.1
        for e in range(self.curEpoch, self.args.epochs+1):
            #记录当前epoch,用于保存Model
            self.curEpoch = e
            log("**************************************************************")
            #训练
            log("start train")
            epoch_loss = self.trainModel()
            log("end train")
            self.train_loss.append(epoch_loss)
            log("epoch %d/%d, epoch_loss=%.2f"% (e, self.args.epochs, epoch_loss))
            
            # if e < 10 and e != 0:
            # else:
            if e < self.args.startTest:
                HR, NDCG = 0, 0
                cvWait = 0
            else:
                HR, NDCG = self.validModel(self.valid_loader)
                log("epoch %d/%d, valid HR = %.4f, valid NDCG = %.4f"%(e, self.args.epochs, HR, NDCG))
            self.his_hr.append(HR)
            self.his_ndcg.append(NDCG)
            

            self.adjust_learning_rate(self.opt, e)
            if HR > best_HR:
                best_HR = HR
                cvWait = 0
                best_epoch = self.curEpoch
                self.saveModel()
            else:
                cvWait += 1
                log("cvWait = %d"%(cvWait))

            self.saveHistory()

            if cvWait == self.args.patience:
                log('Early stopping! best epoch = %d'%(best_epoch))
                self.loadModel(self.modelName)
                break
        
        
    def test(self):
        #load test dataset
        HR, NDCG = self.validModel(self.test_loader)
        log("test HR = %.4f, test NDCG = %.4f"%(HR, NDCG))
        log("model name : %s"%(self.modelName))
    

    def trainModel(self):
        train_loader = self.train_loader
        log("start negative sample...")
        train_loader.dataset.ng_sample()
        log("finish negative sample...")
        epoch_loss = 0
        for user, item_i, item_j in train_loader:
            user = user.long().cuda()
            item_i = item_i.long().cuda()
            item_j = item_j.long().cuda()

            user_embed, item_embed = self.model(self.uv_g)
            
            userEmbed = user_embed[user]
            posEmbed = item_embed[item_i]
            negEmbed = item_embed[item_j]

            pred_i, pred_j = self.innerProduct(userEmbed, posEmbed, negEmbed)

            bprloss = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log().sum()
            regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)

            loss = 0.5*(bprloss + self.args.reg * regLoss)/self.args.batch

            epoch_loss += bprloss.item()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        log("finish train")
        return epoch_loss

    def validModel(self, data_loader, save=False):
        HR, NDCG = [], []
        data = {}
        user_embed, item_embed = self.model(self.uv_g)
        for user, item_i in data_loader:
            user = user.long().cuda()
            item_i = item_i.long().cuda()
            userEmbed = user_embed[user]
            testItemEmbed = item_embed[item_i]
            pred_i = t.sum(t.mul(userEmbed, testItemEmbed), dim=1)

            batch = int(user.cpu().numpy().size/101)
            assert user.cpu().numpy().size % 101 ==0
            for i in range(batch):
                batch_scores = pred_i[i*101: (i+1)*101].view(-1)
                _, indices = t.topk(batch_scores, self.args.top_k)
                tmp_item_i = item_i[i*101: (i+1)*101]
                recommends = t.take(tmp_item_i, indices).cpu().numpy().tolist()
                gt_item = tmp_item_i[0].item()
                HR.append(evaluate.hit(gt_item, recommends))
                NDCG.append(evaluate.ndcg(gt_item, recommends))
        if save:
            return HR, NDCG
        else:
            return np.mean(HR), np.mean(NDCG)


    def getModelName(self):
        title = "KCGN_"
        ModelName = title + self.args.dataset + "_" + modelUTCStr + \
        "_cv" + str(self.args.cv) + \
        "_reg_" + str(self.args.reg)+ \
        "_batch_" + str(self.args.batch) + \
        "_lr_" + str(self.args.lr) + \
        "_decay_" + str(self.args.decay) + \
        "_hide_" + str(self.args.hide_dim) + \
        "_layerNum_" + str(self.args.layerNum) +\
        "_top_" + str(self.args.top_k)

        if self.args.trust == 1:
            ModelName += "_trust"
        return ModelName


    def saveHistory(self):
        #保存历史数据，用于画图
        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        ModelName = self.modelName

        with open(r'./History/' + args.dataset + r'/' + ModelName + '.his', 'wb') as fs:
            pickle.dump(history, fs)

    def saveModel(self):
        # ModelName = self.getModelName()
        ModelName = self.modelName
        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        savePath = r'./Model/' + self.args.dataset + r'/' + ModelName + r'.pth'
        params = {
            'epoch': self.curEpoch,
            'lr': self.lr,
            'model': self.model,
            'reg':self.args.reg,
            'history':history,
            }
        t.save(params, savePath)


    def loadModel(self, modelPath):
        checkpoint = t.load(r'./Model/' + args.dataset + r'/' + modelPath + r'.pth')
        self.curEpoch = checkpoint['epoch'] + 1
        self.lr = checkpoint['lr']
        self.model = checkpoint['model']
        self.args.reg = checkpoint['reg']
        #恢复history
        history = checkpoint['history']
        self.train_loss = history['loss']
        self.his_hr = history['HR']
        self.his_ndcg = history['NDCG']
        log("load model %s in epoch %d"%(modelPath, checkpoint['epoch']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LightGCN main.py')
    #dataset params
    parser.add_argument('--dataset', type=str, default="Yelp", help="Epinions,Yelp,Tianchi")
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--seed', type=int, default=29)

    parser.add_argument('--hide_dim', type=int, default=16)
    parser.add_argument('--layerNum', type=int, default=1)
    parser.add_argument('--trust', type=int, default=0)


    parser.add_argument('--reg', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=0.98)
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--minlr', type=float, default=0.0001)
    parser.add_argument('--test_batch', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--slope', type=float, default=0)
    #early stop params
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_ng', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=10)

    parser.add_argument('--startTest', type=int, default=0)

    args = parser.parse_args()
    print(args)
    args.dataset = args.dataset + "_time"

    mkdir(args.dataset)
    hope = Model(args, isLoadModel)

    modelName = hope.getModelName()
    
    print('ModelNmae = ' + modelName)

    hope.run()
    hope.test()

