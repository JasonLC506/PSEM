import numpy as np
from matplotlib import pyplot as plt

emoticon_labels = ["LOVE", "SAD", "WOW", "HAHA", "ANGRY"]
data_sets = ["CNN", "foxnews", "combine"]
emoticon_dist = [
[9117329, 7161140, 5119423, 8324339, 9208821],
[3289776, 1492824, 1020637, 1738070, 4716509],
[4489780, 2667702, 1762261, 2724278, 5683127]
]

emoticon_dist = np.array(emoticon_dist, dtype=np.float64)
emoticon_dist = emoticon_dist/np.sum(emoticon_dist, axis=-1, keepdims=True)

f, axes = plt.subplots(1,len(data_sets))
for i in range(len(data_sets)):
    axes[i].pie(emoticon_dist[i], labels=emoticon_labels, autopct="%1.1f%%")
    axes[i].set_title(data_sets[i])
plt.legend(bbox_to_anchor=(1.05,0.2))
plt.show()