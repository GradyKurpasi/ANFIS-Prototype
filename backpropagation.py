import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


numIter = 10000

#initialize variables
w1 = .1
w2 = .2
w3 = .3
w4 = .4
w5 = .5
w6 = .6
w7 = .7
w8 = .8
w9 = .9
w10 = .1
wList = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10]

b1 = .5
b2 = .5
bList = [b1, b2]

x1 = 1
x2 = 4
x3 = 5
xList = [x1, x2, x3]

t1 = .1
t2 = .05
tList = [t1, t2]

alpha = .01

def sigmoid(x):
    return np.divide(1, (1 + np.exp(-x)))

def forwardProp(xList, wList, bList):
    print(wList, xList, bList)
    zh1 = wList[0] * xList[0] + wList[2] * xList[1] + wList[4] * xList[2] +bList[0]
    zh2 = wList[1] * xList[0] + wList[3] * xList[1] + wList[5] * xList[2] +bList[0]
    h1 = sigmoid(zh1)
    h2 = sigmoid(zh2)
    print(zh1, zh2, h1, h2)
    zo1 = wList[6] * h1 + wList[8] * h2 + bList[1]
    zo2 = wList[7] * h1 + wList[9] * h2 + bList[1]
    o1 = sigmoid(zo1)
    o2 = sigmoid(zo2)
    print(zo1, zo2, o1, o2)
    return h1, h2, o1, o2

def error(oList, tList):
    return .5 * (np.power(oList[0] - tList[0], 2) + np.power(oList[1] - tList[1], 2))

errList = []

for i in range (numIter):

    h1, h2, o1, o2 = forwardProp(xList, wList, bList)

    sse = error([o1, o2], tList)
    errList.append(sse)

    print('Running ' + str(i + 1) + 'of ' + str(numIter))
    # print('o1: ' + str(o1))
    # print('t1: ' + str(t1))
    # print('o2: ' + str(o2))
    # print('t2: ' + str(t2))
    # print('error: ' + str(sse))
    # print("")

    # Error derivative calcuations
    
    # Compute dE_dw7
    dE_do1 = o1 - t1
    do1_dzo1 = o1 * (1 - o1)
    dzo1_dw7 = h1
    dE_dw7 = dE_do1 * do1_dzo1 * dzo1_dw7

    #Compute dE_dw8
    dE_do2 = o2 - t2
    do2_dzo2 = o2 * (1 - o2)
    dzo2_dw8 = h1
    dE_dw8 = dE_do2 * do2_dzo2 * dzo2_dw8

    #Compute dE_dw9
    dzo1_dw9 = h2
    dE_dw9 = dE_do1 * do1_dzo1 * dzo1_dw9

    #Compute dE_dw10
    dzo2_dw10 = h2
    dE_dw10 = dE_do2 * do2_dzo2 * dzo2_dw10

    #Compute dE_db2 - computes biases for both nodes simultaneously
    dzo1_db2 = 1
    dzo2_db2 = 1
    dE_db2 = dE_do1 * do1_dzo1 * dzo1_db2 + dE_do2 * do2_dzo2 * dzo2_db2

    #Compute dE_dh1 first
    dzo1_dh1 = w7
    dzo2_dh1 = w8
    dE_dh1 = dE_do1 * do1_dzo1 * dzo1_dh1 + dE_do2 * do2_dzo2 * dzo2_dh1

    #Compute dE_dw1
    dh1_dzh1 = h1 * (1 - h1)
    dzh1_dw1 = x1
    dE_dw1 = dE_dh1 * dh1_dzh1 * dzh1_dw1

    #Compute dE_dw3
    dzh1_dw3 = x2
    dE_dw3 = dE_dh1 * dh1_dzh1 * dzh1_dw3

    #Compute dE_dw5
    dzh1_dw5 = x3
    dE_dw5 = dE_dh1 * dh1_dzh1 * dzh1_dw5

    #Compute dE_dw2 first
    dzo1_dh2 = w9
    dzo2_dh2 = w10
    dE_dh2 = dE_do1 * do1_dzo1 * dzo1_dh2 + dE_do2 * do2_dzo2 * dzo2_dh2

    #Compute dE_dw2
    dh2_dzh2 = h2 * (1 - h2)
    dzh2_dw2 = x1
    dE_dw2 = dE_dh2 * dh2_dzh2 * dzh2_dw2

    #Compute dE_dw4
    dzh2_dw4 = x2
    dE_dw4 = dE_dh2 * dh2_dzh2 * dzh2_dw4

    #Compute dE_dw6
    dzh2_dw6 = x3
    dE_dw6 = dE_dh2 * dh2_dzh2 * dzh2_dw6

    #Compute dE_db1
    dzh1_db1 = 1
    dzh2_db1 = 1
    term1 = dE_do1 * do1_dzo1 * dzo1_dh1 * dh1_dzh1 * dzh1_db1
    term2 = dE_do1 * do2_dzo2 * dzo1_dh2 * dh2_dzh2 * dzh2_db1
    dE_db1 = term1 + term2
    
    #Update all parameters
    w1 = w1 - alpha * dE_dw1
    w2 = w2 - alpha * dE_dw2
    w3 = w3 - alpha * dE_dw3
    w4 = w4 - alpha * dE_dw4
    w5 = w5 - alpha * dE_dw5
    w6 = w6 - alpha * dE_dw6
    w7 = w7 - alpha * dE_dw7
    w8 = w8 - alpha * dE_dw8
    w9 = w9 - alpha * dE_dw9
    w10 = w10 - alpha * dE_dw10
    b1 = b1 - alpha * dE_db1
    b2 = b2 - alpha * dE_db2
    wList = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10]
    bList = [b1, b2]
    
# print()
print(wList)

