
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

import os
import numpy as np
import torch
import math
from PIL import Image


def display_cls_confusion_matrix(confusion_mat, labels, test_number,name,method):
    # labels = ['Happiness', 'Sadness', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
    plt.figure(figsize=(9, 7),dpi=300)
    # win_id = self.display_id + 4 if name == 'test' else self.display_id + 5
    color_map = 'Blues' #if name == 'test' else 'Orange'
    confusion_mat = np.array(confusion_mat, dtype=float)
    confusion_mat_number = np.zeros(confusion_mat.shape)
    # test_number = np.array(test_number, dtype=float)
    test_number = np.sum(confusion_mat, axis=1)
    # print('confusion_mat',confusion_mat)
    Expression_count = np.zeros(len(confusion_mat))
    Overall_Accuracy = 0
    
    for i in range(len(confusion_mat)):
        for j in range(len(confusion_mat[i])):
            # print(confusion_mat[i,j], test_number[i])
            confusion_mat[i,j] = confusion_mat[i,j] / test_number[i]
            # print(confusion_mat[i,j], test_number[i])
        confusion_mat_number[i,:] = confusion_mat[i,:] * test_number[i]/100
        for j in range(len(confusion_mat)):
            if(i == j):
                Expression_count[i] += confusion_mat[i, j]
                Overall_Accuracy += confusion_mat[i, i] * test_number[i]
    Overall_Accuracy = Overall_Accuracy / test_number.sum()
    Overall_Accuracy = np.around(Overall_Accuracy, 5)


    UAR = np.around(sum(Expression_count)/len(confusion_mat), 5)
    WAR = Overall_Accuracy

    # print('Expression_count : {}, Overall_Accuracy: {}'.format(Expression_count,Overall_Accuracy))

    # title = 'Confusion Matrix of {} on {} (Accuracy: {}%)'.format(method,name,Overall_Accuracy)
    save_name = '/home/et23-maixj/mxj/JMPF_pooling/plt/'\
                +"Confusion Matrix of %s on %s UAR %s and WAR %s.jpg" % (method,name,UAR,WAR)
    df_cm = pd.DataFrame(confusion_mat, index = labels, columns = labels)
    
    # print('df_cm',df_cm)
    # print('test_number',test_number)
    # print('confusion_mat_number',confusion_mat_number)
    #
    # f, ax = plt.subplots(figsize=(9, 6))
    # ax.set_title(title)
    # ax = sn.heatmap(df_cm, annot=True, cmap=color_map,fmt='.2f',annot_kws={'size':9,'weight':'bold', 'color':'red'})
    ax = sn.heatmap(df_cm, annot=True, cmap=color_map,fmt='.2f',annot_kws={'size':15})

    plt.savefig(save_name, bbox_inches='tight')
    # plt.show()

def display_cls_confusion_matrix_6(confusion_mat, labels, test_number, name, method):
    # 假设中性标签在索引4的位置
    neutral_index = 4

    # 从混淆矩阵中去除中性标签对应的行和列
    confusion_mat = np.delete(confusion_mat, neutral_index, axis=0)  # 删除行
    confusion_mat = np.delete(confusion_mat, neutral_index, axis=1)  # 删除列

    # 从标签列表和测试数量中去除中性标签
    labels = [label for idx, label in enumerate(labels) if idx != neutral_index]
    test_number = [num for idx, num in enumerate(test_number) if idx != neutral_index]

    plt.figure(figsize=(9, 7), dpi=300)
    color_map = 'Blues'
    confusion_mat = np.array(confusion_mat, dtype=float)

    confusion_mat_number = np.zeros(confusion_mat.shape)
    # test_number = np.array(test_number, dtype=float)
    test_number = np.sum(confusion_mat, axis=1)
    # print('confusion_mat',confusion_mat)
    Expression_count = np.zeros(len(confusion_mat))
    Overall_Accuracy = 0
    
    for i in range(len(confusion_mat)):
        for j in range(len(confusion_mat[i])):
            # print(confusion_mat[i,j], test_number[i])
            confusion_mat[i,j] = confusion_mat[i,j] / test_number[i]
            # print(confusion_mat[i,j], test_number[i])
        confusion_mat_number[i,:] = confusion_mat[i,:] * test_number[i]/100
        for j in range(len(confusion_mat)):
            if(i == j):
                Expression_count[i] += confusion_mat[i, j]
                Overall_Accuracy += confusion_mat[i, i] * test_number[i]
    Overall_Accuracy = Overall_Accuracy / test_number.sum()
    Overall_Accuracy = np.around(Overall_Accuracy, 5)


    UAR = np.around(sum(Expression_count)/len(confusion_mat), 5)
    WAR = Overall_Accuracy

    # print('Expression_count : {}, Overall_Accuracy: {}'.format(Expression_count,Overall_Accuracy))

    # title = 'Confusion Matrix of {} on {} (Accuracy: {}%)'.format(method,name,Overall_Accuracy)
    save_name = '/home/et23-maixj/mxj/JMPF_pooling/plt/'\
                +"Confusion Matrix of 10 class %s on %s UAR %s and WAR %s.jpg" % (method,name,UAR,WAR)
    df_cm = pd.DataFrame(confusion_mat, index = labels, columns = labels)
    
    # print('df_cm',df_cm)
    # print('test_number',test_number)
    # print('confusion_mat_number',confusion_mat_number)
    #
    # f, ax = plt.subplots(figsize=(9, 6))
    # ax.set_title(title)
    # ax = sn.heatmap(df_cm, annot=True, cmap=color_map,fmt='.2f',annot_kws={'size':9,'weight':'bold', 'color':'red'})
    ax = sn.heatmap(df_cm, annot=True, cmap=color_map,fmt='.2f',annot_kws={'size':15})

    plt.savefig(save_name, bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':


    # 'FERV39k'  Heatmap
    labels_7 = ['Happiness', 'Sadness', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']

    test_number_7 = [1487, 467, 431, 1473, 1393, 638, 1958]
    name_7 = 'TT (11 classes)'


    confusion_mat_7= np.array([[165,  12,  10,   0,  16,  17,  34],
 [ 18,  39,   2,   4,  10,   9,  12],
 [  6,   3,  53,   0,   0,  22,  38],
 [  4,   6,   1, 139,  58,  11,   6],
 [ 18,   7,   2,   8, 141,   9,  15],
 [  5,   9,   7,   1,   8, 234,  10],
 [ 16,   7,  13,  10,   4,   5, 139]])
    



    display_cls_confusion_matrix(confusion_mat_7,labels_7,test_number_7 , name_7,'IDAA')

