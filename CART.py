"""
decision tree 的代码实现，
CART, ID3, 树的节点划分方法： 信息熵或者基尼指数
quan xueliang
2021.05
"""


import csv
import collections

class DecisionTree:
    " 二叉树，左右分支为正和负 "
    def __init__(self, col=1, value=None, trueBranch=None, falseBranch=None, results=None):
        self.col = col  # 二叉树的叶子节点数
        self.value = value  # 标签，叶子节点的标签
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results


def divideSet(rows, colum, value):
    # 实现对数据集的划分
    '''
    rows: 所有样本， row表示一个样本，是矩阵的一行数据，rows是所有样本，是一个矩阵,含有标签
    colum： 特征维数
    value: 特征值，理解为标签
    '''
    splittingFunction = None

    # 检查数据类型, int 或者 float 类型,现实场景中的数值型特征
    if isinstance(value, int) or isinstance(value, float):
        # python 中高效定义一个函数
        splittingFunction = lambda row: row[colum] >= value
    else: # string数据类型的特征,现实场景中的文本型特征
        splittingFunction = lambda row: row[colum] == value
    # 根据特征的取值，划分到不同的分支（子集）
    list1 = [row for row in rows if splittingFunction(row)]
    list2 = [row for row in rows if not splittingFunction(row)]
    return (list1, list2)  # 为什么是两个子集？因为构建的是二叉树

def uniqueCounts(rows):
    # 函数功能：统计每个特征出现的次数
    " 将输入数据汇总，计算gini值的辅助函数，rows是数据集，输入矩阵形状 [n_samples, d_features]"
    results = {}  # 字典，也可以理解为哈希表
    for row in rows:
        r = row[-1]
        if r not in results: results[r] = 0
        results[r] += 1
    return results
    # 返回的是每个类别的概率，这是一个字典，哈希表


def entropy(rows):
    " 计算数据集的信息熵。"
    from math import log  # 信息熵中有对数运算
    log2 = lambda x: log(x)/log(2)
    # 函数名：log2, 函数输入：x, 函数输出：log(x)/log(2)
    # 信息熵的计算是每个基于每个特征的概率进行的，
    results = uniqueCounts(rows)  # 计算每个类别的概率

    entr = 0.0
    for r in results:
        p = float(results[r])/len(rows)  # 计算每个类别的概率
        entr -= p*log2(p)  # 计算数据集的信息熵
    return entr

def gini(rows):
    " 计算数据集的基尼指数。 "
    # 数据集样本数
    total = len(rows)
    counts = uniqueCounts(rows)  # 统计每个类别的样本数


    # # 数据集基尼指数计算方法1
    # imp = 0.0
    # for k1 in counts:
    #     p1 = float(counts[k1])/total   # 计算类别k1的比例
    #     for k2 in counts:
    #         if k1 == k2: continue
    #         p2 = float(counts[k2])/total
    #         imp += p1*p2
    # return imp

    # 数据集基尼指数计算方法2
    imp = 1.0
    for k in counts:
        p = float(counts[k])/total
        imp -= p*p
    return imp


def variance(rows):
    " 计算模型的方差"
    if len(rows)==0: return 0
    data = [float(row[-1]) for row in rows]  # 所有样本，输出标签
    mean = sum(data)/len(data)

    variance = sum([(d-mean)**2 for d in data]) / len(data)
    return variance


def growDecisionTreeForm(rows, evaluationFunction=entropy):
    """生成并返回一个二叉树，节点分裂使用信息熵或者基尼指数， 输入数据是所有样本，最后一列是标签
    evaluation"""
    if len(rows) == 0: return DecisionTree()
    currentScore = evaluationFunction(rows)  # 计算数据集的信息熵或者基尼指数

    bestGain = 0.0  # 最优增益
    bestAttribute = None  # 最优属性选择
    bestSets = None  # 最优数据子集

    columnCount = len(rows[0]) - 1  # 计算特征维数
    for col in range(columnCount):
        # 所有样本的当前属性值,用于下一步计算当前特征的信息增益
        # 当前维度特征的所有取值，计算以每一个取值为划分的信息增益
        columValues = [row[col] for row in rows]

        # 计算特征的信息增益，进行节点分裂
        # 节点分裂的过程需要有两个for循环，大循环找特征，小循环找特征的分裂值
        for value in columValues:

            (set1, set2) = divideSet(rows, col, value)

            # 计算信息增益，计算以当前特征的取值进行分裂获得的信息增益
            # Gain
            p = float(len(set1)) / len(rows)
            # 计算当前特征当前分裂值的信息增益
            gain = currentScore - p*evaluationFunction(set1) - (1-p)*evaluationFunction(set2)
            # evaluationFunction(set1): 计算左子树的信息熵或基尼指数
            # (1-p)*evaluationFunction(set2)： 计算右子树的信息熵或基尼指数
            # 判断当前特征的当前分裂值能否获得最大的信息增益,同时保证左右子树不能为空
            if gain>bestGain and len(set1)>0 and len(set2)>0:
                bestGain = gain
                bestAttribute = (col, value)
                # col表示特征的索引，也对应者决策树的某一层的节点的属性，value对应树的非叶子节点进行分裂时特征划分的取值
                bestSets = (set1, set2)

    if bestGain > 0:
        # 递归调用生成树的函数本身，生成一棵完整的决策树
        trueBranch = growDecisionTreeForm(bestSets[0])
        falseBranch = growDecisionTreeForm(bestSets[1])
        return DecisionTree(col=bestAttribute[0], value=bestAttribute[1], trueBranch=trueBranch, falseBranch=falseBranch)
    else:
        return DecisionTree(results=uniqueCounts(rows))


"""至此，一棵完整的树就已经生成了，下面的代码主要是对生成的树进行微调，以及调用已经生成的树进行测试"""

def prune(tree, minGain, evaluationFunction=entropy, notify=False):
    """根据最小的信息增益（基于信息熵或者基尼指数）对生成的树进行微调"""
    # 对每个分支进行递归调用
    if tree.trueBranch.results == None: prune(tree.trueBranch, minGain, evaluationFunction, notify)
    if tree.falseBranch.results == None: prune(tree.falseBranch, minGain, evaluationFunction, notify)

    # 树的剪枝，对叶子节点进行合并
    if tree.trueBranch.results!=None and tree.falseBranch.results!=None:
        tb, fb = [], []

        for v, c in tree.trueBranch.results.items(): tb += [[v]]*c
        for v, c in tree.falseBranch.results.items(): fb += [[v]] * c

        p = float(len(tb)) / len(tb+fb)
        delta = evaluationFunction(tb+fb) - p*evaluationFunction(tb) - (1-p)*evaluationFunction(fb)
        # 对树进行剪枝
        if delta < minGain:
            if notify: print('A branch was pruned: gain={:.4f}'.format(delta))
            tree.trueBranch, tree.falseBranch = None, None
            # 如果需要对树进行剪枝，将需要剪枝的叶子节点的数据子集拿掉即可
            # 然后将左右子树的数据子集划分到他们的父节点
            tree.results = uniqueCounts(tb+fb)



def classify(observations, tree, dataMissing=False):
    """
    利用已经生成好的决策树，对测试数据进行预测
    :param observations: 测试样本
    :param tree: 已经学习好的决策树
    :param dartaMissing: 存在数据缺失
    :return: 样本划分到的叶子节点
    """
    # 预测函数1，测试数据不存在缺失值处理
    def classifyWitnoutMissingData(observations, tree):
        if tree.results != None:  # 当前节点是叶子节点, 递归终止条件
            return tree.results

        # 非递归终止条件下，继续进行递归，
        else:
            v = observations[tree.col]  # 从测试样本的最后一个特征开始逐个属性进行判断
            branch = None
            if isinstance(v, int) or isinstance(v, float):  # 数值型特征
                if v >= tree.value: branch = tree.trueBranch
                else: branch = tree.falseBranch
            else:
                if v == tree.value: branch = tree.trueBranch
                else: branch = tree.falseBranch
        return classifyWitnoutMissingData(observations, branch)


    # 预测样本存在缺失值条件下，缺失值的处理方式
    def classifyWithMissingData(observations, tree):
        if tree.results != None:  # 递归到叶子节点，结束递归，返回预测类别
            return tree.results

        else:
            v = observations[tree.col]
            if v == None:  # 如果是缺失值
                # 这里也是递归调用
                tr = classifyWithMissingData(observations, tree.trueBranch)
                fr = classifyWithMissingData(observations, tree.falseBranch)
                tcount = sum(tr.values())
                fcount = sum(tr.values())
                tw = float(tcount) / (tcount+fcount)
                fw = float(fcount) / (tcount+fcount)
                result = collections.defaultdict(int)
                for k, v in tr.items(): result[k] += v*tw
                for k, v in fr.items(): result[k] += v*fw
                return dict(result)
            else:
                branch = None
                if isinstance(v, int) or isinstance(v, float):  #数值型特征
                    if v >= tree.value: branch = tree.trueBranch
                    else: branch = tree.falseBranch
                else:  # 文本型特征
                    if v == tree.value: branch = tree.trueBranch
                    else: branch = tree.falseBranch
            return classifyWithMissingData(observations, branch)


    # function body, 函数主体，根据是否含有缺失值，调用上面的函数进行预测
    if dataMissing:
        return classifyWithMissingData(observations, tree)
    else:
        return classifyWitnoutMissingData(observations, tree)


def plot(decisionTree):
    """输出学习得到的树模型"""
    def toString(decisionTree, indent=''):
        if decisionTree.results != None: # 判断是否是叶子节点
            return str(decisionTree.results)
        else:
            if isinstance(decisionTree.value, int) or isinstance(decisionTree, float):
                # decision = 'column {:.4f}: x >= {:.4f}'.format(decisionTree.col, decisionTree.value)
                decision = 'Column %s: x >= %s?' % (decisionTree.col, decisionTree.value)
            else:
                # decision = 'column {:.4f}: x == {:.4f}'.format(decisionTree.col, decisionTree.value)
                decision = 'Column %s: x == %s?' % (decisionTree.col, decisionTree.value)
            trueBranch = indent + 'yes ->' + toString(decisionTree.trueBranch, indent + '\t\t')
            falseBranch = indent + 'no ->' + toString(decisionTree.falseBranch, indent + '\t\t')
            return (decision + '\n' + trueBranch + '\n' + falseBranch)

    print(toString(decisionTree))  # 打印树模型






def loadCSV(file):
    """
    加载.csv文件，并将所有float和int型数据转换成基本数据类型
    :param file:  文件名称
    :return: 用于生成决策树的数据格式
    """
    def convertTypes(s):
        s = s.strip()
        try:
            return float(s) if '.' in s else int(s)
        except ValueError:
            return s
    reader = csv.reader(open(file, 'rt'))
    return [[convertTypes(item) for item in row] for row in reader]


if __name__ == '__main__':
    example = 1

    if example == 1:
        # 一个很小的数据集
        trainingData = loadCSV('tbc.csv')
        # 节点划分依据基尼指数进行，CART
        decisionTree = growDecisionTreeForm(trainingData, evaluationFunction=gini)  # entropy
        plot(decisionTree)

        print(classify(['ohne', 'leicht', 'Streifen', 'normal', 'normal'], decisionTree, dataMissing=False))
        print(classify([None, 'leicht', None, 'Flocken', 'fiepend'], decisionTree, dataMissing=True))

    else:
        # 稍微大一点点的数据集
        trainingData = loadCSV('fishiris.csv')
        decisionTree = growDecisionTreeForm(trainingData, evaluationFunction=gini)
        plot(decisionTree)

        prune(decisionTree, 0.5, notify=True)
        plot(decisionTree)

        print(classify([6.0, 2.2, 5.0, 1.5], decisionTree))
        print(classify([None, None, None, 1.5], decisionTree, dataMissing=True))


