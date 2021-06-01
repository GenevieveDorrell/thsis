import matplotlib.pyplot as plt
import numpy as np
print(len(["Land Sat Channel 1","Land Sat Channel 2","Land Sat Channel 3","Land Sat Channel 4","Land Sat Channel 5","Land Sat Channel 6",
                    "Land Sat Channel 7", "Land Sat Channel 8", "Tree Cover", "Loss Year", "Slope", "Elevation", "Aspect", "Stream Distance", "Stand Gge", "Stand Age 80", "Stand Age 200"]))

feature_names = ["Land Sat Channel 1","Land Sat Channel 2","Land Sat Channel 3","Land Sat Channel 4","Land Sat Channel 5","Land Sat Channel 6",
                    "Land Sat Channel 7", "Land Sat Channel 8", "Tree Cover", "Loss Year", "Slope", "Elevation", "Aspect", "Stream Distance", "Stand Gge", "Stand Age 80", "Stand Age 200"]
averaged_importance = []
for i in range(17):
    averaged_importance.append(i)
plt.figure(figsize=(20,10))
plt.bar(feature_names, averaged_importance,color = 'mediumspringgreen')
plt.title("Feature importance of RF",fontsize=44)
plt.legend(fontsize=36,frameon=False)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().tick_params(labelsize=24,width=2)
plt.xticks(rotation=45,horizontalalignment='right')
plt.savefig("RF_feature_importance.png",bbox_inches="tight")