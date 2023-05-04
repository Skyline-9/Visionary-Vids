import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# INSERT DATA HERE
pred_probs = np.random.rand(60)  # X-axis
gt_probs = np.random.rand(60)  # X-axis
time = np.arange(pred_probs.shape[0])  # Y-axis

fig, ax = plt.subplots(figsize=(16, 2))

# plot the lines
ax.plot(time, pred_probs, label="Predicted", color="blue")
ax.plot(time, gt_probs, label="Ground Truth", color="red")

# add plot features (ex. legend, axis ticks, etc.)
# ax.legend()
ax.legend(loc="upper right", bbox_to_anchor=(1.13, 1))


ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

# add labels
plt.xlabel("Time")
plt.ylabel("Saliency Score")

# show plot
plt.show()

# save the plot
plt.savefig("qvhighlights_plot.png")
