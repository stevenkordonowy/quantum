import numpy as np
import csv 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg
import sys
import seaborn as sns
import pandas as pd
import math as math

def print_goodies(data, best, p):
  e = 0.999 * best
  idxs = data[:,-1] >= e
  near_max = data[idxs]
  near_max = near_max[near_max[:,-1].argsort()]

  for row in near_max:
    print('{}'.format(*row))

  # ax.scatter(near_max[:,0], near_max[:,1], near_max[:,2], marker='X', label='near')

def plot2(name):
  print('###plo2({})'.format(name))
  fig = plt.figure()

  data = []
  with open('{}.csv'.format(name), 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)
    for row in reader:
      # normalize the z-axis to make a % of happy vertices
      data.append((float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])/11))
  
  data = np.array(data)
  # data = data[data[:,4].argsort()]

  idx_best = np.argmax(data[:,-1])
  best = data[idx_best]
  print('best[{}]: {}'.format(idx_best, best))
  
  ax = fig.add_subplot(121, projection='3d')
  ax.scatter(data[:,0], data[:,1], data[:,4], marker='.', label=str(name))
  print_goodies(data,best[4], 2)
  ax.set_xlabel('gamma1')
  ax.set_ylabel('beta1')
  ax.set_zlabel('% happy vertices')
  ax.set_title('{0} vertices'.format(name))

  ax = fig.add_subplot(122, projection='3d')
  ax.scatter(data[:,2], data[:,3], data[:,4], marker='.', label=str(name))
  # print_goodies(data,best,2)
  ax.set_xlabel('gamma2')
  ax.set_ylabel('beta2')
  ax.set_zlabel('% happy vertices')
  ax.set_title('{0} vertices'.format(name))
  
  plt.legend(loc='upper left')
  plt.show()
  # plt.savefig(name)

def plot1(name):
  print('###plo1({})'.format(name))
  fig = plt.figure()

  data = []
  with open('{}.csv'.format(name), 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)
    for row in reader:
      # normalize the z-axis to make a % of happy vertices
      data.append((float(row[0]), float(row[1]), float(row[2])/7))
  
  data = np.array(data)

  # sns.set(style = "darkgrid")
  idx_best = np.argmax(data[:,-1])
  best = data[idx_best]
  print('best: {}'.format(best))
  
  ax = fig.add_subplot(111, projection='3d')
  cmhot = plt.get_cmap("BuPu")
  ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,2], cmap=cmhot)
  # print_goodies(data,best[2], 1)
  ax.set_xlabel(r'$\gamma$', fontsize=18)
  ax.set_ylabel(r'$\beta$', fontsize=18)
  ax.set_zlabel('Happy vertices (%)')
  # ax.set_title(r'$M(C)$ for $p=1$ on $2-$regular graphs', pad=20)

  ax.set_xticks([0, math.pi/4, math.pi/2, 3 * math.pi/4, math.pi, 5*math.pi/4, 3 * math.pi/2, 7 * math.pi/4, 2*math.pi])
  ax.set_xticklabels([0, '', '', '', r'$\pi$', '', '', '', r'2$\pi$'])

  ax.set_yticks([0, math.pi/8, 2*math.pi/8, 3 * math.pi/8, math.pi/2, 5*math.pi/8, 6 * math.pi/8, 7 * math.pi/8, math.pi])
  ax.set_yticklabels([0, '', '', '', r'$\pi$/2', '', '', '', r'$\pi$'])
  
  # plt.legend(loc='upper left')
  plt.show()
  plt.savefig(name)


def plot(name, p):
  print('name={},p={}'.format(name,p))
  if p == 2:
    plot2(name)
  elif p == 1:
    plot1(name)
  else:
    print('cannot handle p={}'.format(p))


if __name__ == "__main__":
  args = sys.argv[1:]

  print('args:{}'.format(args))
  plot1(args[0])
  # plot2(args[0])

  # plot(args[0], int(args[1]))

