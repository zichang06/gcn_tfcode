import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))  #  strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
    return index  # cora, len(index) = 1000


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    ## 对cora数据集：
    # x     scipy.sparse.csr.csr_matrix (140, 1433)
    # y     numpy.ndarray               (140, 7)
    # tx    scipy.sparse.csr.csr_matrix (1000, 1433)
    # ty    numpy.ndarray               (1000, 7)
    # allx  scipy.sparse.csr.csr_matrix (1708, 1433)
    # ally  numpy.ndarray               (1708, 7)
    # graph collections.defaultdict
    # graph是一个字典,len=2708,大图总共2708个节点
    # for i in graph:
    #     print(i,graph[i])

    # 测试数据集的索引乱序版
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))  # 测试数据集的索引乱序版
    test_idx_range = np.sort(test_idx_reorder)  # 从小到大排序,如[1707,1708,1709,...2707]

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    # 将allx和tx叠起来并转化成LIL格式的feature,即输入一张整图
    features = sp.vstack((allx, tx)).tolil()  # (2708,1433),LTL格式
    # 把特征矩阵还原为其原本节点，和对应的邻接矩阵对应起来，因为之前是打乱的，不对齐的话，特征就和对应的节点搞错了。
    # 比如测试集tx(1708-2707)的第0个节点，是原图的第2692号节点。
    # ???那除测试集1000个外的其它节点呢？不需要变下么？因为allx没有乱序？
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # 邻接矩阵格式也是LIL的，并且shape为(2708, 2708)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):  # 把sparse_mx list中的每个稀疏矩阵都转成
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()  
            # type(mx):scipy.sparse.coo.coo_matrix,将csr_matrix转成coo_matrix
            # scipy.sparse.coo_matrix - A sparse matrix in COOrdinate format.
        coords = np.vstack((mx.row, mx.col)).transpose()   # Stack arrays in sequence vertically (row wise). 一行接一行.
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx  # type:tuple

# 处理特征:特征矩阵进行归一化,变成COO,并返回一个格式为(coords, values, shape)的元组
# 特征矩阵的每一行的每个元素除以行和，处理后的每一行元素之和为1，正则化输入特征
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    # type(features):lil_matrix
    # a.sum()是将矩阵中所有的元素进行求和;a.sum(axis = 0)是每一列列相加;a.sum(axis = 1)是每一行相加
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    # type(feature):scipy.sparse.lil.lil_matrix
    # type(r_mat_inv):scipy.sparse.dia.dia_matrix
    features = r_mat_inv.dot(features)
    # type(features):scipy.sparse.csr.csr_matrix
    return sparse_to_tuple(features)

# 邻接矩阵adj对称归一化并返回coo存储模式
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)  # adj的类型由csr_matrix 转变为 coo_matrix.  为啥要转变???,上面features还是lil呢，计算速度更快？
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # np.power是element-wise
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^(-1/2)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # (AD)^(T)D = D^(T)AD


# 将邻接矩阵加上自环以后，对称归一化，并存储为COO模式，最后返回元组格式
def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))  # A += I, D^(T)AD
    return sparse_to_tuple(adj_normalized)

# 构建输入字典并返回
# Python 字典(Dictionary) dict.update(dict2) 函数把字典dict2的键/值对更新到dict里。
#labels和labels_mask传入的是具体的值，例如
# labels=y_train,labels_mask=train_mask；
# labels=y_val,labels_mask=val_mask；
# labels=y_test,labels_mask=test_mask；
def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    # features[1].shape: (49216,); features[0].shape: (49216,2), type(features[1].shape):tuple
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)  # D^(T)AD  (D → D^(-1/2))    type:COO
    laplacian = sp.eye(adj.shape[0]) - adj_normalized   # L^sys = D^(T)LD = I - D^(T)AD  type:CSR
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')  # 获取L^sys最大的特征值
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])  # 式(5)下第一行 L~  type:CSR

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))  # I  第0阶  scipy.sparse.dia.dia_matrix  
    t_k.append(scaled_laplacian)  # L  第1阶  CSR

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)  # 好像原来也是CSR
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two  # Tk(L)=2LTk-1(L)-Tk-2(L)

    for i in range(2, k+1):  # 2~k阶
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
