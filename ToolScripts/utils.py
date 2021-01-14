import pickle as pk
from ToolScripts.TimeLogger import log
import torch as t
import scipy.sparse as sp
import numpy as np
import os
import networkx as nx

def mkdir(dataset):
    DIR = os.path.join(os.getcwd(), "History", dataset)
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    DIR = os.path.join(os.getcwd(), "Model", dataset)
    if not os.path.exists(DIR):
        os.makedirs(DIR)

def matDropOut(mat, rate):
    assert rate < 1.0
    log("mat nnz = %d"%(mat.nnz))
    row_idx, col_idx = mat.nonzero()
    nums = int(mat.nnz * rate)
    idx = np.random.permutation(row_idx.shape[0])[: nums]
    res = sp.csr_matrix((np.ones_like(row_idx[idx]), (row_idx[idx], col_idx[idx])), shape=mat.shape)
    res = (res + sp.eye(mat.shape[0]) != 0) *1
    assert res.max() == 1
    log("mat nnz after dropout= %d"%(res.nnz))
    return res

def matExpand(uuMat, rate=0.001):
    # rate = 0.001
    log("expand rate = %.4f"%(rate))
    row, col = uuMat.shape
    for i in range(row):
        tmpMat = (sp.random(1, col, density=rate, format='csr') != 0) * 1
        if i == 0:
            res = tmpMat
        else:
            res = sp.vstack((res, tmpMat))
    res2 = res + uuMat
    res2 = (res2 != 0) * 1
    log("expand count = %d"%(res2.nnz-uuMat.nnz))
    return res


def get_neigbors(graph, node, depth=1):
    output = {}
    layers = dict(nx.bfs_successors(graph, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1, depth+1):
        output[i] = []
        for x in nodes:
            # output[i].extend(layers.get(x,[]))
            output[i] += layers.get(x, [])
        nodes = output[i]
    return output

def buildSubGraph(args, mat, subNode):
    node_num = mat.shape[0]
    graph = nx.Graph(mat)

    # k_sub_graph = {}
    # for node in range(node_num):
    #     res = get_neigbors(graph, node, 4)
    #     k_sub_graph[node] = res
    #     log('%d/%d'%(node, node_num), save=False, oneline=True)

    # DIR = os.path.join(os.path.dirname(os.getcwd()), "dataset", args.dataset+"_time", 'implicit', "cv{0}".format(args.cv))
    # with open(DIR + '/k_sub_graph.pkl', 'wb') as fs:
    #     pk.dump(k_sub_graph, fs)

    subGraphList = list(nx.connected_components(graph))
    subGraphCount = len(subGraphList)
    node_subGraph = [-1 for i in range(node_num)]
    adjMat = sp.dok_matrix((subGraphCount, node_num), dtype=np.int)
    node_list = []
    for i in range(len(subGraphList)):
        subGraphID = i
        subGraph = subGraphList[i]
        if len(subGraph) > subNode:
            node_list += list(subGraph)
        for node_id in subGraph:
            assert node_subGraph[node_id] == -1
            node_subGraph[node_id] = subGraphID
            adjMat[subGraphID, node_id] = 1
    node_subGraph = np.array(node_subGraph)
    assert np.sum(node_subGraph == -1) == 0 
    adjMat = adjMat.tocsr()
    return subGraphList, node_subGraph, adjMat, node_list

def loadData2(datasetStr, cv):
    assert datasetStr == "Tianchi_time"
    DIR = os.path.join(os.path.dirname(os.getcwd()), "dataset", datasetStr, 'implicit', "cv{0}".format(cv))
    with open(DIR + '/pvTime.pkl', 'rb') as fs:
        pvTimeMat = pk.load(fs)
    with open(DIR + '/cartTime.pkl', 'rb') as fs:
        cartTimeMat = pk.load(fs)
    with open(DIR + '/favTime.pkl', 'rb') as fs:
        favTimeMat = pk.load(fs)
    with open(DIR + '/buyTime.pkl', 'rb') as fs:
        buyTimeMat = pk.load(fs)
    interatctMat = ((pvTimeMat + cartTimeMat + favTimeMat + buyTimeMat) != 0) * 1
    interatctMat = interatctMat.astype(np.bool)
    return interatctMat
    


def loadData(datasetStr, cv):
    if datasetStr == "Tianchi_time":
        return loadData2(datasetStr, cv)
    DIR = os.path.join(os.path.dirname(os.getcwd()), "dataset", datasetStr, 'implicit', "cv{0}".format(cv))
    log(DIR)
    with open(DIR + '/train.pkl', 'rb') as fs:
        trainMat = pk.load(fs)
    return trainMat

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if type(sparse_mx) != sp.coo_matrix:
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = t.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = t.from_numpy(sparse_mx.data)
    shape = t.Size(sparse_mx.shape)
    return t.sparse.FloatTensor(indices, values, shape)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()

def generate_sp_ont_hot(num):
    mat = sp.eye(num)
    # mat = sp.dok_matrix((num, num))
    # for i in range(num):
    #     mat[i,i] = 1
    ret = sparse_mx_to_torch_sparse_tensor(mat)
    return ret

def load(path):
    with open(path, 'rb') as fs:
        data = pk.load(fs)
    return data



    