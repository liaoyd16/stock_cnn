import csv

import os
TICKS = "/Users/liaoyuanda/Desktop/模式识别/大作业/Tick/"
JSON = "/Users/liaoyuanda/Desktop/模式识别/大作业/TickJson/"
filelist = os.listdir(TICKS)

import matplotlib.pyplot as plt
import json

import numpy as np

INF = 1e10

# :blank
# 9:15 - 9:25   : 0-10 mins
# 9:25 - 11:30  : 10-135 mins
# 13:00 - 15:00 : 225-345 mins

K = 0.003
EPS = 1e-5

BLANK = 15
MORNING = 135
NOON = 225
AFTERNOON = 345

TIMEBIN = 20

START = int(np.floor(BLANK / TIMEBIN))
BREAK = int(np.ceil(MORNING / TIMEBIN))
RECOVER = int(np.floor(NOON / TIMEBIN))
END = int(np.ceil(AFTERNOON / TIMEBIN))

print(START, BREAK, RECOVER, END)

def smooth(entry, dt):
    k = 1-np.exp(- dt * K)
    if int(entry[3]) == 0:
        ans = int(entry[2])
    else:
        ans = k * 1000 * int(entry[4])/int(entry[3]) + (1-k) * int(entry[2])
    return ans

def __minute(timestring):
    return (int(timestring[:-7]) - 9) * 60\
         + int(timestring[-7:-5])\
         + int(timestring[-5:-3]) / 60\
         + int(timestring[-3:]) / 3600\
         - 15

def __mins(timestring):
    return int(timestring[-7:-5])

def __hours(timestring):
    return (int(timestring[:-7]) - 9)

global cnt
cnt = 0
def get_vectors_in_csv(csvname):
    global cnt
    print(cnt, "processing {}".format(csvname))
    cnt += 1
    csvrd = csv.reader(open(TICKS + csvname + ".csv", "r"))
    for line in csvrd: break
    csvdata = [line for line in csvrd]
    if (len(csvdata) == 0): return
    i = 0
    j = 0
    vectors = []

    if len(csvdata) == 0: return

    timestamp = 0

    while i < len(csvdata):
        while j < len(csvdata) and (__minute(csvdata[j][1]) - timestamp) < TIMEBIN:
            j += 1
        j -= 1
        # print(i, j, __minute(csvdata[i][1]), __minute(csvdata[j][1]), timestamp)

        if j == i - 1:
            # print(__minute(csvdata[j+1][1]) - timestamp)
            zero_intervals = int((__minute(csvdata[j+1][1]) - timestamp) // TIMEBIN)
            for k in range(zero_intervals):
                vectors.append([timestamp + k*TIMEBIN, 0, 0, 0, 0, 0, 0, 0])
            timestamp += zero_intervals * TIMEBIN

        else:
            segment = csvdata[i:j+1]
            prices = [smooth(segment[ientry], __minute(segment[ientry+1][1]) - __minute(segment[ientry][1]) ) for ientry in range(j - i - 1)]
            prices.append(float(segment[-1][2]))
            prices = np.array(prices)

            prices = np.delete(prices, np.where(prices == 0)[0])
            if len(prices) > 0:
                P0 = prices[0]
                Pt = prices[-1]
                P_high = max(prices)
                P_low  = min(prices)
                P_average = np.mean(prices)
                # timestamp
                volume   = int(segment[-1][9]) - int(segment[0][9])
                turnover = int(segment[-1][10]) - int(segment[0][10])

                vectors.append([timestamp, P0, P_low, P_high, Pt, P_average, volume, turnover])

            timestamp += TIMEBIN

        j += 1
        i = j

    if len(vectors) == 0: return

    vectors_export = vectors[START:BREAK]
    vectors_export.extend(vectors[RECOVER:])
    if len(vectors) < BREAK - START + END - RECOVER:
        vectors_export.extend([np.zeros(8) for _ in range(BREAK - START + END - RECOVER - len(vectors))])
    vectors_export = np.array(vectors_export)

    min_ = np.min(vectors_export[:,2])
    max_ = np.max(vectors_export[:,3])
    vectors_export[:,1:6] = (vectors_export[:,1:6] - min_) / (max_ - min_+EPS)
    min_ = np.min(vectors_export[:,6])
    max_ = np.max(vectors_export[:,6])
    vectors_export[:,6]   = (vectors_export[:,6] - min_) / (max_ - min_+EPS)
    min_ = np.min(vectors_export[:,7])
    max_ = np.max(vectors_export[:,7])
    vectors_export[:,7]   = (vectors_export[:,7] - min_) / (max_ - min_+EPS)

    plt.plot(vectors_export[:,0], vectors_export[:,1:5])
    plt.show()

    if not os.path.isdir(JSON):
        os.mkdir(JSON)
    json.dump(vectors_export.tolist(), open(JSON + csvname + ".json", "w"))

for file in filelist:
    if ".csv" in file:
        get_vectors_in_csv(file[:-4])

# get_vectors_in_csv("Tick_33949_850")
