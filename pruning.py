
import numpy as np,re
import random
import Image, ImageDraw
import operator
from scipy.cluster.vq import kmeans,vq
from numpy import linalg as LA
from pylab import plot,show,setp,gca,getp

CUTOFF_THRESHOLD = -80
KNN = 20
PIXEL2METER = 17.2
NUMCLUSTER = 7

# '02e04c7673f9', '34bdc8db1ec0', '20aa4b5e2019' lower than -80
# '00258422fcd3', '00258422fcd4' , '00258422fcd2', '00258422fcd1', '00258422fcd3', '00258422fcd4'
CORRUPTAPS = ['001fca5d49b0', '00258422fcd5', '00258422fcd0']
CORRUPTFACTOR = 0.6
SAMPLESIZE = 130
UPDATEFACTOR = 0.5
UPDATECIRCLE = 1.0

# WL = r'C:\Users\kris\Documents\Visual Studio 2012\Projects\cosine\Release\wl.txt'
REF = r'D:\Software Install Document\Dropbox\WiFi Fingerprinting\Journal Sim\experiment\ust chameleon\wifi data\LTJ reference ever small\p_all_LTJ.txt'
# REF = r'D:\Software Install Document\Dropbox\WiFi Fingerprinting\Journal Sim\experiment\ust chameleon\wifi data\LTJ reference small\p_all_LTJ.txt'
TAR = r'D:\Software Install Document\Dropbox\WiFi Fingerprinting\Journal Sim\experiment\ust chameleon\wifi data\LTJ target 2\p_target_all.txt'
IMAGEURL = r'D:\Software Install Document\Dropbox\WiFi Fingerprinting\Journal Sim\experiment\ust chameleon\survey map\2.png'
OUTIMAGEDIR = r'D:\Software Install Document\Dropbox\WiFi Fingerprinting\Journal Sim\experiment\ust chameleon\wifi data\AP Pruning\o_'
ERRORDIR = r'D:\Software Install Document\Dropbox\WiFi Fingerprinting\Journal Sim\experiment\ust chameleon\wifi data\LTJ target 2\err_'


def getRefLineDetails(line,cutoffThres):
    items = line.split(' ',1)
    loc = [int(i) for i in items[0].split(',')]
    rssVec = dict((i[0].lower(), float(i[1])) for i in re.findall(r'(\w+):(.*?),', items[1]) if float(i[1])>cutoffThres and i[0].lower())
    return loc,rssVec
def getRefData(refFile,cutoffThres):
    lines = [l for l in open(refFile).readlines() if ',' in l]
    result = [getRefLineDetails(i,cutoffThres) for i in lines]
    macSet = set(i[0].lower() for i in re.findall(r'(\w+):(.*?),',open(refFile).read()) if float(i[1])>cutoffThres)
    print 'REF\t%s: %d points found, %d macs found'%(refFile,len(result),len(macSet))
    return macSet,result
def getTarLineDetails(line, refMacSet, cutoffThres):
    items = line.split(' ', 1)
    loc = [int(i) for i in items[0].split(',')]
    rssVec = dict((i[0].lower(), float(i[1])) for i in re.findall(r'(\w+):(.*?),', items[1]) if float(i[1])>cutoffThres and i[0].lower() in refMacSet)
    return loc,rssVec
def getTarData(tarFile,refMacSet,cutoffThres):
    lines = [l for l in open(tarFile).readlines() if ',' in l]
    result = [getTarLineDetails(i,refMacSet,cutoffThres) for i in lines]
    print 'TAR\t%s: %d points found'%(tarFile,len(result))
    return result

def calcEuclideanSim(refVec,tarVec,a=2.0,b=10.0):
    size = len(tarVec)
    v1=[0.0]*size
    v2=[0.0]*size
    for i, did in enumerate(tarVec.keys()):
        if did in refVec: v1[i] = float(refVec[did])
        if did in tarVec: v2[i] = float(tarVec[did])
    # print 'v1 and v2 are ', v1, v2
    sim = np.sqrt(sum(np.subtract(v1, v2)**2))
    # sim = np.dot(v1,v2)/LA.norm(v1,2)/LA.norm(v2,2)
    # if np.isnan(sim): return 0.0
    # else: return sim
    return sim

# some error of order of corruption
def calcSim(refVec,tarVec,a=2.0,b=10.0):
    size = len(tarVec)
    v1=[0.0]*size
    v2=[0.0]*size
    for i, did in enumerate(tarVec.keys()):
        if did in refVec: v1[i] = a**(float(refVec[did])*1.0/b)
        if did in tarVec: v2[i] = a**(float(tarVec[did])*1.0/b)
    # print 'v1 and v2 are ', v1, v2
    sim = np.dot(v1,v2)/LA.norm(v1,2)/LA.norm(v2,2)
    if np.isnan(sim): return 0.0
    else: return sim

def dist(p1,p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

def final_selection(sims,idx,points_ref,n=KNN):
    finalGroup = [i for i in idx[-1*n:]]
    finalSims = [sims[i] for i in finalGroup]
    Normalizer = sum(1/((1-min(sims[i],0.9999))**2) for i in finalGroup)
    if Normalizer<10e-3: return 0,0
    x,y = 0,0
    for i in finalGroup:
        x += 1/((1-min(sims[i],0.9999))**2)/Normalizer*points_ref[i][0]
        y += 1/((1-min(sims[i],0.9999))**2)/Normalizer*points_ref[i][1]
    return x,y

def get_location(refData, singleTar):
    refPts = [i[0] for i in refData]
    sims = []
    for j in xrange(len(refData)):
        sims.append(calcSim(refData[j][1],singleTar))
    idx = sorted(range(len(sims)), key=lambda x: sims[x])
    x,y = final_selection(sims,idx,refPts)

    return [x, y]

def calc_max_sim(refData, singleTar):
    sims = []
    for j in xrange(len(refData)):
        sims.append(calcSim(refData[j][1],singleTar))

    return max(sims)

def cos_test(refFile,tarFile):
    macSet,refData = getRefData(refFile,CUTOFF_THRESHOLD)
    if len(refData)==0: return
    refPts = [i[0] for i in refData]
    tarData = getTarData(tarFile,macSet,CUTOFF_THRESHOLD)
    if len(tarData)==0: return

    tarData = corrupt_tar(tarData)

    err = []
    for i in xrange(len(tarData)):
        loc = tarData[i][0]
        sims = []
        for j in xrange(len(refData)):
            sims.append(calcSim(refData[j][1],tarData[i][1]))
        idx = sorted(range(len(sims)), key=lambda x: sims[x])
        x,y = final_selection(sims,idx,refPts)
        e = dist([x,y],loc)
        err.append(e)
    print '%f\t%f'%(np.mean(err)/PIXEL2METER, np.std(err)/PIXEL2METER)
    write_error_to_file(err, r'cos_est_corrupt.txt')
    # for i in err: print i

def euc_test(refFile,tarFile):
    macSet,refData = getRefData(refFile,CUTOFF_THRESHOLD)
    if len(refData)==0: return
    refPts = [i[0] for i in refData]
    tarData = getTarData(tarFile,macSet,CUTOFF_THRESHOLD)
    if len(tarData)==0: return
    err = []
    for i in xrange(len(tarData)):
        loc = tarData[i][0]
        sims = []
        for j in xrange(len(refData)):
            sims.append(calcEuclideanSim(refData[j][1],tarData[i][1]))
        idx = sorted(range(len(sims)), reverse=True, key=lambda x: sims[x])
        # idx = sorted(range(len(sims)), key=lambda x: sims[x])
        x,y = final_selection(sims,idx,refPts)
        e = dist([x,y],loc)
        err.append(e)
    print '%f\t%f'%(np.mean(err)/PIXEL2METER, np.std(err)/PIXEL2METER)
    # for i in err: print i

def get_virtual_rss(tarVec):
    macDict = {}
    for mac in tarVec.keys():
        if random.random() > float(0.5):
            macDict[mac] = tarVec[mac]

    return macDict

def corrupt_tar(tarData):
    for singleTar in tarData:
        for mac in singleTar[1].keys():
            if mac in CORRUPTAPS:
                singleTar[1][mac] *= CORRUPTFACTOR
    return tarData

def draw_locations(locations):
    im = Image.open(IMAGEURL)
    draw = ImageDraw.Draw(im)
    for loc in locations:
        draw.ellipse((loc[0], loc[1], loc[0]+10, loc[1]+10), fill='red')
    # im.show()
    im.save(OUTIMAGEDIR+r'tarOne.png')

def plot_clusters(locations, idx, centroids):
    locations = np.array(locations)
    print 'the cluster labels ', idx
    plot(locations[idx==0,0], locations[idx==0,1], 'ob',
         locations[idx==1,0], locations[idx==1,1], 'or',
         locations[idx==2,0], locations[idx==2,1], 'ok',
         locations[idx==3,0], locations[idx==3,1], 'oc',
         locations[idx==4,0], locations[idx==4,1], 'oy',
         locations[idx==5,0], locations[idx==5,1], 'om',
         locations[idx==6,0], locations[idx==6,1], 'og',)
    plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
    setp(gca(), 'ylim', reversed(getp(gca(), 'ylim')))
    show()

def make_cluster(locations):
    locations = np.array(locations)
    centroids,_ = kmeans(locations, NUMCLUSTER) #labeled from 0
    idx, distance = vq(locations,centroids)

    return idx, distance, centroids 

def calc_prune_ratio(macsPruned, clusterLabels, tarVec):
    # find the prune list in each cluster
    macsPrunedAllCluster = []
    for i in range(NUMCLUSTER):
        macsPrunedEachCluster = []
        for j in range(len(clusterLabels)):
            if clusterLabels[j] == i:
                macsPrunedEachCluster.append(macsPruned[j]) #something wrong here
        macsPrunedAllCluster.append(macsPrunedEachCluster)

    # calc prune ratio
    macsPruneRatioAllCluster = []
    for clusterIndex in range(len(macsPrunedAllCluster)):
        macPruneDict = {}
        for mac in tarVec.keys():
            tempCounter = 0
            for macsEachSample in macsPrunedAllCluster[clusterIndex]:
                if mac in macsEachSample:
                    tempCounter += 1
            if len(macsPrunedAllCluster[clusterIndex]) > 0:
                macPruneDict[mac] = tempCounter/float(len(macsPrunedAllCluster[clusterIndex]))
            else:
                macPruneDict[mac] = float(0)
        macsPruneRatioAllCluster.append(macPruneDict)
    # print 'final prune ratio', macsPruneRatioAllCluster
    return macsPruneRatioAllCluster

def pick_good_cluster(locations, clusterLabels, toCentroidDis, macsPrunedAllSample, macsPrunedRatioAllCluster):
    # select clusters with max common prune ratio > 0.5
    # then take the centroid location & nearest centroid RSS & pruned ap
    goodClusters = []
    for numCluster, macsPrunedRatioEachCluster in enumerate(macsPrunedRatioAllCluster):
        if max(macsPrunedRatioEachCluster) > float(0.5):
            goodClusters.append(numCluster)

    nearestCentroidPrunedAPs = []
    nearestCentroidLocs = []
    for clusterIdx in goodClusters: #find the nearest centroid pruned aps and estimated locations
        minIdxEachCluster = 10000
        minDisEachCluster = float(10000)
        for idx in range(len(clusterLabels)):
            if clusterLabels[idx] == clusterIdx:
                if toCentroidDis[idx] < minDisEachCluster:
                    minDisEachCluster = toCentroidDis[idx]
                    minIdxEachCluster = idx
        nearestCentroidPrunedAPs.append(macsPrunedAllSample[minIdxEachCluster])
        nearestCentroidLocs.append(locations[minIdxEachCluster])
    # print 'macs pruned all samples', macsPrunedAllSample
    # print 'clustering locations ', locations
    # print 'clustering labels', clusterLabels
    # print 'clustering distance to centroid', toCentroidDis
    # print 'the rss and locations near the centroid', nearestCentroidLocs, nearestCentroidPrunedAPs
    return nearestCentroidLocs, nearestCentroidPrunedAPs

def select_final_good_cluster(oneTar, nearestCentroidLocs, nearestCentroidPrunedAPs, refData):
    # calcluate the final good clusters
    clusterSims = []
    for eachCentroidPruneSet in nearestCentroidPrunedAPs:
        tempCentroidRss = {}
        for eachAP in list(set(oneTar.keys()) - set(eachCentroidPruneSet)):
            tempCentroidRss[eachAP] = oneTar[eachAP]
        maxSim = calc_max_sim(refData, tempCentroidRss)
        clusterSims.append(maxSim)
    maxIdx, maxValue = max(enumerate(clusterSims), key=operator.itemgetter(1))
    print 'sim of all clusters', clusterSims
    print 'cluster index and its similarity', maxIdx, maxValue

    print 'final good result', nearestCentroidLocs[maxIdx]
    return nearestCentroidLocs[maxIdx]

def find_nearest_ref(locationEst, refData):
    tempDis = []
    for ref in refData:
        distance = dist(locationEst, [ref[0][0],ref[0][1]])
        tempDis.append(distance)
    minIdx, minValue = min(enumerate(tempDis), key=operator.itemgetter(1))

    return minIdx, minValue

def update_radio_map(tarVec, locationEst, refData):
    # another is to constrain it to 2m circle
    nearestRefIdx, nearestDis = find_nearest_ref(locationEst, refData)
    if nearestDis/PIXEL2METER < UPDATECIRCLE:
        for mac in refData[nearestRefIdx][1].keys():
            if mac in tarVec.keys():
                refData[nearestRefIdx][1][mac] = \
                    UPDATEFACTOR*refData[nearestRefIdx][1][mac] + (1-UPDATEFACTOR)*tarVec[mac]

    return refData

def write_error_to_file(error, estType):
    outfile = open(ERRORDIR + estType, 'w')
    for item in error:
        outfile.write('%s '%item)
    outfile.write('\n')
    outfile.close()

def pruning(refFile, tarFile):
    # Given target and reference,
    macSet,refData = getRefData(refFile,CUTOFF_THRESHOLD)
    if len(refData)==0: return
    refPts = [i[0] for i in refData]
    tarData = getTarData(tarFile,macSet,CUTOFF_THRESHOLD)
    if len(tarData)==0: return

    tarData = corrupt_tar(tarData) #corrupt data then calc its 
    
    error = []
    for oneTar in tarData:
        trueLoc = [oneTar[0][0], oneTar[0][1]]
        locations = []
        macsPrunedAllSample = []
        # get virtual locations and pruned aps
        for i in range(SAMPLESIZE):
            virtualVec = get_virtual_rss(oneTar[1])
            if len(virtualVec.keys()) > 3:
                locations.append(get_location(refData, virtualVec))
                # macsPruned.append(virtualVec.keys())
                macsPrunedAllSample.append(list(set(oneTar[1].keys()) - set(virtualVec.keys())))
        # do clustering and get final estimation
        clusterLabels, toCentroidDis, centroids = make_cluster(locations)
        macsPruneRatioAllCluster = calc_prune_ratio(macsPrunedAllSample, clusterLabels, oneTar[1])
        nearestCentroidLocs, nearestCentroidPrunedAPs = pick_good_cluster(locations, clusterLabels, toCentroidDis, macsPrunedAllSample, macsPruneRatioAllCluster)
        # final selection based on similarity with radio map
        finalEst = select_final_good_cluster(oneTar[1], nearestCentroidLocs, nearestCentroidPrunedAPs, refData)
        # update the radio map
        refData = update_radio_map(oneTar[1], finalEst, refData)
        e = dist(finalEst, trueLoc)
        error.append(e)
    print '%f\t%f'%(np.mean(error)/PIXEL2METER, np.std(error)/PIXEL2METER)
    write_error_to_file(error, r'pruning_tar2_1circle.txt')


def pruning_test(refFile, tarFile):
    # Given target and reference,
    macSet,refData = getRefData(refFile,CUTOFF_THRESHOLD)
    if len(refData)==0: return
    refPts = [i[0] for i in refData]
    tarData = getTarData(tarFile,macSet,CUTOFF_THRESHOLD)
    if len(tarData)==0: return

    tarData = corrupt_tar(tarData) #corrupt data then calc its 
    singleTar = tarData[5]
    # 1) For one tar RSSI, get all virtual RSSs and get its estimated loction
    oneTar = singleTar[1]
    print 'target true location', singleTar[0]
    locations = []
    macsPrunedAllSample = []
    for i in range(SAMPLESIZE):
        virtualVec = get_virtual_rss(oneTar)
        if len(virtualVec.keys()) > 3:
            locations.append(get_location(refData, virtualVec))
            # macsPruned.append(virtualVec.keys())
            macsPrunedAllSample.append(list(set(oneTar.keys()) - set(virtualVec.keys())))
    # print 'all macs', oneTar.keys()
    # for pruned in macsPruned:
    #     print sorted(pruned)
    # draw_locations(locations)
    clusterLabels, toCentroidDis, centroids = make_cluster(locations)
    # print 'cluster labels and distance ', clusterLabels, toCentroidDis
    macsPruneRatioAllCluster = calc_prune_ratio(macsPrunedAllSample, clusterLabels, oneTar)
    nearestCentroidLocs, nearestCentroidPrunedAPs = pick_good_cluster(locations, clusterLabels, toCentroidDis, macsPrunedAllSample, macsPruneRatioAllCluster)
    # final selection based on similarity with radio map
    finalEst = select_final_good_cluster(oneTar, nearestCentroidLocs, nearestCentroidPrunedAPs, refData)
    print 'target true location ', singleTar[0]
    draw_locations(locations)
    plot_clusters(locations, clusterLabels, centroids)


pruning(REF,TAR)

# pruning_test(REF, TAR)

# cos_test(REF, TAR)
