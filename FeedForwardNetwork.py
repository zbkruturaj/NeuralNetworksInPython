# CBOW - HirarchicalSoftmax

import numpy as np
import csv
import operator

freqTable = {}

with open('single_word_counts.csv', 'rb') as csv_file:
    reader = csv.reader(csv_file)
    freqTable = dict(reader)



freq = freqTable.items()
htList = []
for item in freq:
    htList.append(item)
heapq.heapify(htlist)
i = 0
while len(freq)>1:
    ht1 = heapq.heappop(htList)
    ht2 = heapq.heappop(htList)
    ht
