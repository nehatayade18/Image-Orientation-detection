# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 12:26:16 2019

@author: Neha
"""

import sys
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelBinarizer
import pickle
import math
from queue import PriorityQueue


class NeuralNet(object):
    def __init__(self, iterations=5000, learnRate=0.15 , layers=[192, 10, 4], lamb=0.5, keep_probab=0.8):
        self.set_params(iterations, learnRate, lamb, layers, keep_probab)
        
    def set_params(self, iterations, learnRate, lamb, layers, keep_probab):
        self.iterations = iterations
        self.learnRate = learnRate
        self.lamb = lamb
        self.layers = layers
        self.keep_probab = keep_probab
        self.initialization(layers)
    #https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/
    def initialization(self, layers):
        self.params = {}
        for i in range(1, len(layers)):
            step = str(i)
            self.params["w"+str(i)] = np.random.randn(layers[i],layers[i-1]) * 0.1
            #print(self.params["w"+str(i)])
            self.params["b"+str(i)] = np.zeros((layers[i],1))

    #Activation functions used: relu,Sigmoid,softmax
    def Relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    #Output within range 0 to 1
    def softmax(self, x):
        x_exp = np.exp(x - np.max(x))
        exp_sum = np.sum(x_exp,  axis=0, keepdims=True)
        return x_exp / exp_sum
    
    #Dropout nodes
    #https://towardsdatascience.com/coding-neural-network-dropout-3095632d25ce
    #https://stackoverflow.com/questions/54109617/implementing-dropout-from-scratch
    def dropout(self, x):
        drop = np.random.rand(x.shape[0], x.shape[1])
        drop = drop < self.keep_probab
        x *= (drop/self.keep_probab)
        return x, drop#droupout output masked and scaled
    
    def crossentropy(self, a_out, Z, Y):
        m = np.log(np.sum(np.exp(Z)+1e-4, axis=0, keepdims=True))
        return - np.sum(np.multiply(Y+1e-4, (Z - m)))
    
    def cost(self, a_out, Y, ops):
        Z = ops[-1][1]
        tot_w = 0
        for i in range(1,len(self.layers)-1):
            w_ = np.square(self.params["w"+str(i)])
            tot_w += np.sum(w_)
        regularized = (tot_w * self.lamb) / (2 * Y.shape[1])
        cross_entropy =  self.crossentropy(a_out, Z, Y) / Y.shape[1]
        cross_entropy =  np.squeeze(cross_entropy)
        total_cost = cross_entropy + regularized
        return total_cost
    
    #Referred to for forwar&backward: https://towardsdatascience.com/neural-networks-from-scratch-easy-vs-hard-b26ddc2e89c7
    def forward_activation(self,a_initial,w,b,act):
        print(w.shape)
        print(a_initial.shape)
        Z = np.dot(w, a_initial) + b
        if act == 'softmax':
            a = self.softmax(Z)
            op = ((a_initial, w, b), Z)
        else:
            methods = ['Relu']
            act = np.random.choice(methods)
            if act == 'Relu':
                a = self.Relu(Z)
                a, d = self.dropout(a)
                op = ((a_initial, w, b), Z, d)
            elif act == 'sigmoid':
                a = self.sigmoid(Z)
                op = ((a_initial, w, b), Z)
        return a, op
        
    def forward_propagation(self, a):
        ops = [] 
        for i in range(1,len(self.layers)-1):
            step = str(i)
            w = self.params["w"+str(i)]
            b = self.params["b"+str(i)]
            a_initial = a
            method = ['Relu','sigmoid']
            act = np.random.choice(method)
            a, op = self.forward_activation(a_initial, w, b, act)
            ops.append(op)

        w = self.params["w" + str(len(self.layers)-1)]
        b = self.params["b" + str(len(self.layers)-1)]
        a_out, op = self.forward_activation(a, w, b, "softmax")
        ops.append(op)

        return a_out, ops
    
    
    def backward_activation(self, dZ, op, layer):
        a_initial, w, b = op
        w = self.params["w" + str(layer)]
        da_initial = np.dot(w.T, dZ)
        s = a_initial.shape[1]
        dw = np.dot(dZ, a_initial.T) + (self.lamb * w) / s * (1/s) 
        db = (1/s) * np.sum(dZ, axis=1, keepdims=True)
        return da_initial, dw, db
    
    def backward_recursive(self, layer, da, op, act):
        #print(op)
        ((a_initial, w, b), Z, d) = op

        if act == "der_Relu":
            da = da * d/ self.keep_probab
            dZ = np.array(da, copy=True)
            dZ[Z <= 0] = 0

        elif act == "der_softmax":
            dZ = self.softmax(Z) - Z
            
        da_initial, dw, db = self.backward_activation(dZ, (a_initial, w, b), layer)

        return da_initial, dw, db
    #https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html
    def backward_propagation(self, a_out, Y, ops):
        doh = {}
        op = ops[len(ops)-1]
        dZ = a_out - Y.reshape(a_out.shape)
        layer = len(ops)
        a,b,c = self.backward_activation(dZ, op[0], layer)
        doh["da"+str(layer)], doh["dw"+str(layer)], doh["db"+str(layer)] = a, b, c
        for l in range(layer-1, 0, -1):
            op = ops[l-1]
            da = doh["da"+str(l+1)]
            d, e, f = self.backward_recursive(l, da, op, "der_Relu")
            doh["da" + str(l)] = d
            doh["dw" + str(l)] = e
            doh["db" + str(l)] = f
        
        return doh
    
    def nnet_train(self, X_train, y_train):
        cost_proportion = [(0,0.0)]
        for i in range(self.iterations):
            a_out, ops = self.forward_propagation(X_train)
            doh = self.backward_propagation(a_out, y_train, ops)
            calculated_cost = self.cost(a_out, y_train, ops)
            #Updating parameters
            for l in range(1, len(self.layers)):
                self.params["w"+str(l)] -= self.learnRate * doh["dw"+str(l)]
                self.params["b"+str(l)] -= self.learnRate * doh["db"+str(l)]
            X, Y = X_train, y_train
            #Batch gradient descent implemented here
            if i%200 == 0:
                if i % 1000 == 0:
                    self.learnRate /= 2
                cost_proportion.append((i, calculated_cost))
                pred, acc = self.test(X, Y)
                #print(y_test.shape)
                #print(a_out.shape)
                #correct = np.mean(actual == predict)
                #print(actual,predict)
                print("Iteration", i, "->", "Accuracy", acc)
                
    def nnet_test(self, X_test, y_test):
        X_test, y_test = X_test, Y_test
        for layer in range(1, len(self.layers)-1):
            w = self.params["w" + str(layer)]
            b = self.params["b" + str(layer)]
            #print(w.shape)
            #print(X_test.shape)
            Z = np.dot(w, X_test) + b
            a = self.Relu(Z)

        w = self.params["w" + str(len(self.layers)-1)]
        b = self.params["b" + str(len(self.layers)-1)]
        Z = np.dot(w, a) + b
        a_out = self.softmax(Z)

        actual = np.argmax(y_test,axis=0)#y_test.argmax(0)
        predict = np.argmax(a_out,axis=0)#a_out.argmax(0)
        correct = len(actual) - np.count_nonzero(actual - predict)
        return predict, round(correct / len(actual), 4) * 100
    
if __name__ == "__main__":
    task =  sys.argv[0]
    filename = sys.argv[1]
    model_file = sys.argv[2]
    model = sys.argv[3]

    if task == "train":
        #trainset = pd.read_table(filename, sep=" ", header=None)
        #images = trainset.iloc[:, 0]
        #x_train = trainset.iloc[:, 2:]
        #y_train = trainset.iloc[:, 1]
        #X_train = x_train.T
        #Y_train = y_train.T
        images = np.loadtxt(filename,usecols=0,dtype=str)
        X_train = np.loadtxt(filename,usecols=range(2,194),dtype=int)
        Y_train = np.loadtxt(filename,usecols=1,dtype=int)
        shuffle_indices = range(len(Y_train))
        np.random.shuffle(X_train)
        np.random.shuffle(Y_train)
        #print(Y_train)
        if model == "nnet":
            L = LabelBinarizer()
            Ly = L.fit(Y_train)
            Y_train = Ly.transform(Y_train)
            iterations = 5000
            learnRate = 0.15
            layers = [X_train.shape[1],10,Y_train.shape[1]]
            #print(layers)
            lamb = 0.9
            keep_probab = 0.8
            neural_net = NeuralNet(iterations=5000, learnRate=learnRate,layers= layers, lamb= lamb, keep_probab= keep_probab)
            neural_net.nnet_train(X_train.T/int(255), Y_train.T)
            models = (Ly, neural_net)
        model_save = open(model_file, 'wb')
        pickle.dump(models, model_save)
            
    elif task == 'test':
        #testset = pd.read_table(filename, sep=" ", header=None)
        #images = testset.iloc[:, 0]
        #X_test = testset.iloc[:, 2:]
        #Y_test = testset.iloc[:, 1]
        images = np.loadtxt(filename,usecols=0,dtype=str)
        X_test = np.loadtxt(filename,usecols=range(2,194),dtype=int)
        Y_test = np.loadtxt(filename,usecols=1,dtype=int)
        shuffle_indices = range(len(Y_test))
        np.random.shuffle(X_test)
        np.random.shuffle(Y_test)
        models = pickle.load(model_file)
        if model == 'nnet':
            y, neural_net = models
            Y_test = y.transform(Y_test)
            pred, score = neural_net.nnet_test(X_test.T, Y_test.T)
            output_file = open('output.txt', 'w')
            for image, pred in zip(images, pred):
                output_file.write( '\n' + str(image) + ' ' + str(pred))
#reference: https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/
#https://adventuresinmachinelearning.com/stochastic-gradient-descent/

        elif model == "K-nearest":
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
    if classifier.lower() == "K-nearest":
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
        

