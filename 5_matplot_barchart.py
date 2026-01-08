"""Multiple Bar Plots example (fixed indentation and style)

Run: python .\5_matplot_barchart.py
"""

import numpy as np
import matplotlib.pyplot as plt

barWidth = 0.25
fig, ax = plt.subplots(figsize=(10, 6))

IT = [12, 30, 1, 8, 22]
ECE = [28, 6, 16, 5, 10]
CSE = [29, 3, 24, 25, 17]

br1 = np.arange(len(IT))
br2 = br1 + barWidth
br3 = br2 + barWidth

ax.bar(br1, IT, color='r', width=barWidth, edgecolor='grey', label='IT')
ax.bar(br2, ECE, color='g', width=barWidth, edgecolor='grey', label='ECE')
ax.bar(br3, CSE, color='b', width=barWidth, edgecolor='grey', label='CSE')

ax.set_xlabel('Branch', fontweight='bold', fontsize=15)
ax.set_ylabel('Students passed', fontweight='bold', fontsize=15)
ax.set_xticks([r + barWidth for r in range(len(IT))])
ax.set_xticklabels(['2015', '2016', '2017', '2018', '2019'])
ax.legend()

plt.tight_layout()
plt.show()