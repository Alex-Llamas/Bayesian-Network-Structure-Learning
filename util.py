import math


def maxlist(a, b):
    result = []
    for i in range(0, len(a)):
        result.append(max(a[i], b[i]))
    return result


def getOrderRank(xlist):
    rank = [i[0] for i in sorted(enumerate(xlist), key=lambda x: x[1])]
    rank.reverse()
    return rank


def maxPosition(l):
    position = 0
    maxVal = 0
    for i in range(len(l)):
        if (l[i] > maxVal):
            maxVal = l[i]
            position = i
    return position


def normalize(arr):
    maxValue = max(arr)
    arr[:] = [round(x / maxValue, 2) for x in arr]
    return arr


#################################### Split sizes ####################################

def splitSize(size, split, zero=True):
    split = int(split)
    bins = [size] * split
    binsSize = float(size) / split
    for i in range(0, split):
        bins[i] = int(round(binsSize * (i)))
        if (bins[i] == 1):
            bins[i] = 2
    bins.append(size)
    if (zero == False):
        bins.remove(0)
    bins = list(set(bins))
    bins.sort()
    return bins


def intervalSplit(infLim, supLim, split=2, limitIncluded=False):
    bins = []
    print "[", infLim, ",", supLim, "]"
    if (limitIncluded):
        stepSize = (supLim - infLim) / (split - 1.0)
        addValue = infLim
        while (addValue <= supLim):
            bins.append(int(round(addValue)))
            addValue = addValue + stepSize
    else:
        stepSize = (supLim - infLim) / (split + 1.0)
        addValue = infLim
        while (addValue + stepSize < supLim):
            addValue = addValue + stepSize
            bins.append(int(round(addValue)))
    return bins


#################################### Compute steps ####################################
def computeStep(rangeX, rangeY, N, v=2, sigma=0.95):  # useSteps = 0
    num = rangeX * rangeY * v * sigma
    den = N * pow((1 - sigma), 0.5)
    step = pow(num / den, 0.5)
    resultX = int(math.floor(rangeX / step))
    resultY = int(math.floor(rangeY / step))
    if (resultX <= 2):
        resultX = 2
    if (resultY <= 2):
        resultY = 2
    return [resultX, resultY]


def computeStepNormalized(N, v=2, sigma=0.95):  # useSteps = 2
    return int(math.floor(1 / float(math.sqrt((v * sigma) / (N * math.sqrt(1 - sigma))))))


def computeStepV2(rangeData, N, v=2, sigma=0.95):  # useSteps = 2
    if (rangeData > (pow(N, 0.5) / 2)):
        num = rangeData * v * sigma
        den = N * pow((1 - sigma), 0.5)
        step = pow(num / den, 0.5)
        result = int(math.floor(rangeData / step))
    else:
        result = rangeData
    return int(result)


#################################### DATASET TYPE ANALYSIS ####################################

def datesetType(y):  # useSteps = 2
    samples = len(y)
    domainTarget = len(list(set(y)))
    if (domainTarget <= pow(samples,
                            0.5) / 2):  # eg. 100 samples must have at most sqrt(100)/2 = 5 differnt target to be consider classification problem
        return 0  # classification
    else:
        return 1  # Regression

    if (rangeData > (pow(N, 0.5) / 2)):
        num = rangeData * v * sigma
        den = N * pow((1 - sigma), 0.5)
        step = pow(num / den, 0.5)
        result = int(math.floor(rangeData / step))
    else:
        result = rangeData
    return int(result)


#################################### VOTE SELECTION ####################################

def sumMixedCorrelation(ll, normalize=False):
    ll = normalizeScales(ll)
    arr = [round(sum(x) / len(ll), 2) for x in zip(*ll)]
    if (normalize):
        maxValue = max(arr)
        return [round(x / maxValue, 2) for x in arr]
    else:
        return arr


def findMultiplier(ll):
    w = [max(sublist) for sublist in ll]
    w = [max(w)] * len(w)
    i = 0
    for l in ll:
        w[i] = round(float(w[i]) / max(l), 2)
        i = i + 1
    return w


def normalizeScales(ll):
    w = findMultiplier(ll)
    for i in range(len(ll)):
        for j in range(len(ll[i])):
            ll[i][j] = round(ll[i][j] * w[i], 2)
    return ll
