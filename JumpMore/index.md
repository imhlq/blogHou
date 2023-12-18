---
title: "Jumpmore 跳一跳刷分机器"
date: 2018-01-14
tags: ["游戏", "研究"]
heroImage: "71kKj.gif"
draft: false
---

## 简介
腾讯的微信为致敬（抄袭）育碧而推出一款叫跳一跳的小游戏。操作简单、界面简洁，很适合做分析。刚开始看到这款游戏，就隐隐中觉得很像Flappy Bird。于是很自然的想到，可以用Flappy Bird同样的办法刷分。

但随后发现，这款游戏由于回合离散性，可以用更简单的办法。下面简述过程。

## 0. 环境搭建

初始试图提取微信网页，但微信认证机制较为复杂。
因此为了简单起见。本次采用Python编写脚本，利用ADB Platform Tools调试连接。
手机为三星s8, 2220\*1080像素

## 1. 思路步骤

大致思路：
1. 利用ADB抓取手机屏幕
2. 使用程序分析屏幕，并做基本图像处理
3. 分析当前坐标及下一步目标坐标
4. 计算得出按压時间
5. 利用ADB模拟按压

> 本思路参考[moneyDboat](https://github.com/moneyDboat/wechat_jump_jump)

## 2. 历程

当前位置分析较为容易，由于角色始终不变，可使用模式识别直接获取坐标，效果非常好，几乎可做到无误差识别。
```python
img = cv2.imread(pic)
cimg = cv2.imread(character_path)
csize = cimg.shape[:2]
res = cv2.matchTemplate(img, cimg, cv2.TM_SQDIFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
```


为了获取下一步坐标，参考代码使用圆点为主、边缘为辅的手段。但该手段并不适用于当前版本跳一跳。首先、微信跳一跳已更新新版本，取消了大部分方块中间的圆点、小部分则只在按下之后出现圆点。其次，采取边缘获取的方法仅适用与亮色简单方块（圆块），对于新出现的药罐、卷纸等物体毫无作用，导致游戏结束。

**为此，务必使用全新的分析方式。**

首先尝试依旧采取物体边缘计算物体中心点位置，并对特殊情况做修正。故利用OpenCV对截图做边缘化处理。
```Python
img = cv2.imread(pic)   # Read image from file
img = cv2.GaussianBlur(img, (5, 5), 0)   # do GaussianBlur to decrease noise and extra boundary
img_canny = cv2.Canny(img, 1, 10)   # Do canny
```
![jump_nn-1](https://xhou.me/content/images/2018/01/jump_nn-1.png)
利用边缘化后图像，获取下一方块顶部坐标：（代码仅适用2220*1080屏幕）
```
# Get Top position
y_top = np.nonzero([max(x[:-200]) for x in img_canny[600:]])[0][0] + 600
x_top = int(np.mean(np.nonzero(img_canny[y_top][:-200])))
```

中心点X坐标与顶部方块X坐标非常靠近，因此，可以直接获得中心X坐标。
Y方向坐标初始采用检测下边缘方法确定。但在尝试中发现问题，如前文所言：仅适用与亮色简单方块（圆块），对于新出现的药罐、卷纸等物体毫无作用。于是对部分方块进行修正。修正方法为抓取方块特定列像素点进行分析。
但由于修正难度较大，方块类型较多，并不能取得很好的拓展性，方法因为依赖运气（是否碰到特殊方块），分数可刷到300分左右。

进一步，发现边缘获取法对最大边缘的获取准确度较高。因此不妨尝试获取方块顶部边缘与地面边缘。然后通过特定比例取得中心点坐标。
```python
nextPo = []
for i in range(y_top+5, y_top + 350):
    if img_canny[i][x_top] != 0:
        nextPo.append(i)
print(nextPo, len(nextPo))
y_bottom = y_top + (nextPo[-1] - y_top) / 1.9
```
该方法适用性较好，首次尝试，已经可以将分数推到500分左右。

但分析失败原因，仍是由于特定方块的特殊形状，使得方块的身高比不固定。使得部分中心点选取到靠近边缘导致掉落。

因此，对特定方块误差做一级修正。

一级修正的方法有：
1. 像素点结构分析
2. 像素点数分析
3. 像素点结构与数量混合分析
4. 机器学习模块识别修正

### 2.2 像素点结构数量混合修正
下面代码对特定药瓶、长筒卷纸等方块进行修正，主要根据像素点数量与像素点间距等简单特征识别，并作出经验修正。
```python
nextPo = []
    for i in range(y_top+5, y_top + 350):
        if img_canny[i][x_top] != 0:
            nextPo.append(i)
#     y_bottom = y_bottom - 70
    #     print('Long disk')
    # elif len(nextPo) < 4 and nextPo[-1] - nextPo[0] > 150:
    #     y_bottom = y_bottom + 25
    #     print('simple fix')
    # elif len(nextPo) < 4 and nextPo[-1] - nextPo[0] < 150:
    #     y_bottom = y_bottom - 20
    #     print('simple fix')
    # elif 15 < len(nextPo) < 23:
    #     y_bottom = y_bottom - 25
    #     print('drug')
```
该方法可将跳一跳分数刷到700分左右。

第二周初，对该方法优化后，仅利用像素数目对特定方块修正，即达到800+分。
```python
if 4 < len(nextPo) < 11:
    # sp for long cylinder
    y_bottom = y_bottom - 30
    print('Long disk!')
elif 100 < len(nextPo) < 150:
    y_bottom = y_bottom - 50
    print('O type')
elif 13 < len(nextPo) < 22:
    y_bottom = y_bottom - 50
    print('drug')
```

### 2.3 机器学习模块识别修正
该方法基于Flappy Bird的机器学习AI算法。
鉴于闭源游戏在時间上与多线程上的局限性，难以对跳一跳做直接的强化学习过程。退而求其次，可以利用人工辅助采集数据，机器学习数据的方法进行。

人工采集代码如下，即利用前文脚本进行游戏，并记录方块特征。人工记录每一步中误差大小储存为pickle文件。

```python
def createmldata():
    data = []
    while True:
        pullScreen()
        p1 = getCharacterPosition(screen_path)

        img = cv2.imread(screen_path)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img_canny = cv2.Canny(img, 1, 10)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(p1[2])
        for k in range(min_loc[1] - 10, min_loc[1] + 159):
            for b in range(min_loc[0] - 10, min_loc[0] + 80):
                img_canny[k][b] = 0

        y_top = np.nonzero([max(x[:-200]) for x in img_canny[600:]])[0][0] + 600
        x_top = int(np.mean(np.nonzero(img_canny[y_top][:-200])))

        nextblock = []
        for yi in range(y_top-10, y_top+230):
            nextblock.append(np.copy(img_canny[yi, x_top - 120: x_top + 120]))

        nextPo = []
        for i in range(y_top + 5, y_top + 350):
            if img_canny[i][x_top] != 0:
                nextPo.append(i)

        y_bottom = y_top + (nextPo[-1] - y_top) / 1.9
        x_bottom = x_top
        x_center = int((x_top + x_bottom) / 2)
        y_center = int((y_top + y_bottom) / 2)

        cv2.rectangle(img_canny,
                      (x_center - 10, y_center - 10),
                      (x_center + 10, y_center + 10),
                      (255, 0, 0),
                      4
                      )

        nextblockimg = np.array(nextblock, np.int32)
        cv2.imwrite('jump_ml.png', img_canny)

        err = int(input('Please Input Position Error: '))
        if err == 404:
            exit()
        acc = y_center - y_top - err

        data.append((nextblockimg, acc))
        # save
        fr = open('data.ml', 'wb')
        pickle.dump(data, fr)
        fr.close()
        print('Ok!')
        loop()
        time.sleep(2)
```

之后利用卷积神经网络，输入为像素特征，输出为（对应类型的）误差修正。本程序使用了3层神经网络。首先将图像转化为240\*240像素点进入输入层，然后进行5x5的卷积层进行卷积操作与2x2池化操作。之后利用Dropout层处理过拟合问题，并进入输出层。
学习过程采用速度为0.001的AdamOptimizer方法。*(框架基于Tensorflow)*

```python
# Get Center point from ML way
# 2018-01-12

import tensorflow as tf
import cv2
import pickle
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def createNetwork():
    # 80*80 -> 32*32 -> 32*32 -> 1*1
    x_image = tf.placeholder("float", [None, 240, 240, 1])

    # Layer 1
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Dense Layer
    W_d = weight_variable([60 * 60 * 32, 512])
    b_d = bias_variable([512])
    h_pool1_flat = tf.reshape(h_pool1, [-1, 60 * 60 * 32])
    h_out = tf.nn.relu(tf.matmul(h_pool1_flat, W_d) + b_d)

    # Dropout Layer
    keep_prob = tf.placeholder(tf.float32)
    h_drop = tf.nn.dropout(h_out, keep_prob=keep_prob)

    # Readout
    W_out = weight_variable([512, 1])
    b_out = bias_variable([1])
    y_out = tf.matmul(h_drop, W_out) + b_out

    return x_image, keep_prob, y_out


def trainNetwork(x_image, y_out, keep_prob, sess):
    # loss
    y_ = tf.placeholder(tf.float32)
    loss = tf.reduce_mean(tf.square(y_out - y_))
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    sess.run(tf.global_variables_initializer())
    # start Training
    t = 0
    tdata = pickle.load(open('data.ml', 'rb'))
    ximgs = []

    ximg = np.reshape(tdata[t][0], (240, 240, 1))
    ximgs.append(ximg)
    ylabel = [x[1] for x in tdata]

    train_step.run(feed_dict={y_: ylabel, x_image: ximgs, keep_prob: 0.5})
    print('Train Finished')

    cost = loss.eval(feed_dict={
        y_: ylabel, x_image: ximgs, keep_prob: 1.0})
    print('training error %g' % cost)


def learn():
    sess = tf.InteractiveSession()
    x_img, keep_prob, y_out = createNetwork()
    trainNetwork(x_img, y_out, keep_prob, sess)


learn()
```
对于收集的数据，可以做镜像、位移等数据扩展方法。
之后对源程序步骤中加入预测误差结果，可取的较好效果。

更多细节与内容，请参阅源代码[Github](https://github.com/imhlq/wechat_jumpmore).


## 最新思路

在构思本文時，突然发现新的思路。利用游戏特性的几何规律

可直接通过A点横坐标与人物位置确定C点坐标。(感谢JYlsc提供思路)


但是该方法存在误差积累问题，即每回合下一步的位置基于上一步人物位置。当人物位置产生误差時，误差会传递到下一回合。

## All in all
~~因此目前为止仍无完美的解决方案~~，但本文所说方法刷到1000分然后被朋友删好友当无任何问题。