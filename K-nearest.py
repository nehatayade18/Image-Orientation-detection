# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 19:21:48 2019

@author: sachath,ntayade
"""
import sys
import math
from queue import PriorityQueue
import numpy as np


def processing():

    lineNum = 0
    classifier = open(YFile, "r")
    for row in classifier:
        rowList = row.split('|')
        intOrientation = rowList[0]
        intVector = rowList[1].split(' ')
        vector = [int(i) for i in intVector]
        trainOrientation[lineNum] = intOrientation
        trainVector[lineNum] = np.array(vector)
        lineNum += 1
    classifier.close()
    lineNum = 0
    testData = open(XFile, "r")
    for line in testData:
        testList = line.split(' ')
        testFile[lineNum] = testList[0]
        testList = [int(i) for i in testList[1:]]
        testOrientation[lineNum] = testList[0]
        testVector[lineNum] = np.array(testList[1:])
        lineNum += 1
    testData.close()
    
def training(trainFile, YFile):
    trainData = open(trainFile, "r")
    YAppend = open(YFile, "w")
    for line in trainData:
        row = line[:-1].split(' ', 2)
        YAppend.write("%s|%s\n" % (row[1], row[2]))
    trainData.close()
    YAppend.close()   
    
#Calculating euclidean distance and making prediction with neighbors

def testing(testVector, trainOrientation, trainVector, kValue, angle, knnDist):
    distQueue = PriorityQueue()
    for row in range(0,len(trainOrientation),1):
        vector = trainVector[row]
        orient = int(trainOrientation[row])
        eucDistance = math.sqrt(np.sum(np.power((vector - testVector), 2)))
        distQueue.put((eucDistance, orient))
        #print eucDistance
    k=kValue
    for i in range(0, k, 1):
        knnScore = distQueue.get()
        angle[knnScore[1]] += 1
        knnDist[knnScore[1]] += knnScore[0]
    prediction = max(angle, key=angle.get)
     #print prediction
    return prediction   
   
dataset = sys.argv[1]
XFile = sys.argv[2]
YFile = sys.argv[3]
classifier = sys.argv[4]
if classifier.lower() == "nearest":
    if dataset.lower() == "train":
        training(XFile, YFile)
    if dataset.lower() == "test":
        output = open("output.txt", "w+")
        accuracy = 0
        numLinesTrain = sum(1 for line in open(YFile))
        numLinesTest = sum(1 for line in open(XFile))
        trainVector = np.zeros((numLinesTrain, 192),dtype= int)
        trainOrientation = np.zeros((numLinesTrain, 1),dtype=int)
        testVector = np.zeros((numLinesTest, 192),dtype=int)
        testOrientation = np.zeros((numLinesTest, 1),dtype=int)
        testFile = np.empty(numLinesTest, dtype='S256')
        processing()
        for row in range(0, len(testOrientation), 1):
            angle = {0: 0, 90: 0, 180: 0, 270: 0}
            knnDist = {0: 0, 90: 0, 180: 0, 270: 0}
            prediction = testing(testVector[row], trainOrientation, trainVector, 40, angle, knnDist)
            output.write("%s %s\n" % (str(testFile[row]), str(prediction)))
            # accuracy calculation
            accuracy += (1 if prediction == int(testOrientation[row]) else 0)
        print ("Accuracy: " + str(100.0 * accuracy / row))
        output.close()

