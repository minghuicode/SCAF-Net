from collections import defaultdict
from math import sqrt
import random
from time import time
import json
import numpy as np


class Point(object):
    x = None
    y = None
    def __init__(self, x,y):
        self.x = x
        self.y = y
    def x(self):
        return self.x
    def y(self):
        return self.y
    def __add__(self,other):
        return Point(self.x + other.x, self.y + other.y)
    def __div__(self,value):
        return Point(self.x/float(value),self.y/float(value))
    def __truediv__(self,value):
        return self.__div__(value)
    def dist(self,other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    def closest(self,points):
        return min(points,key=self.dist)
    def __repr__(self):
        return "(%f,%f)" % (self.x,self.y)

def update_centroids(points,centroids):
    groups = groupby(points,centroids)
    res = []
    for g in groups.values():
        res.append(sum(g, Point(0,0))/len(g))
    return res

def groupby(points,centroids):
    g = defaultdict(list)
    for p in points:
        c = p.closest(centroids)
        g[c].append(p)
    return g

def show(centroids):
    rv = []
    for p in centroids:
        x,y = p.x, p.y
        i,j = int(round(np.exp(x))), int(round(np.exp(y)))
        rv.append((i,j))
    print(rv)


def run(xs, n, iters=15):
    # centroids = xs[:n]
    centroids = random.sample(xs,n)
    for i in range(iters):
        centroids = update_centroids(xs,centroids)
    # print(centroids)
    show(centroids)
    return groupby(xs,centroids)

def getpoints():
    with open('boxes.json','r') as fp:
        src_d = json.load(fp)
    small = []
    large = []
    total = []
    for short in src_d:
        for x,y, w,h in src_d[short]:
            if w+h<1: continue
            p = np.log(w)
            q = np.log(h)
            if w<40 and h<40:
                small.append(Point(p,q))
            else:
                large.append(Point(p,q))
            total.append(Point(p,q))
    return small, large, total

def kmeans(points,n=5):
    run(points,n)



if __name__ == "__main__":
    smalls, larges, total = getpoints()
    print('smalls: ',len(smalls), '  larges: ',len(larges))
    print('getting small anchors ...')
    for i in [3]:
        kmeans(smalls,i)
    print('getting large anchors ...')
    for i in [3]:
        kmeans(larges,i)
    print('getting total anchors ...')
    for i in [6]:
        kmeans(total,i)
