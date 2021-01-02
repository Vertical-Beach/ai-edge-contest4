
import pickle
import matplotlib.pyplot as plt
import sys
import numpy as np

def draw_loss_curve(t_loss, v_loss, path):
	t = np.arange(0, len(t_loss))
	plt.figure(figsize=(8,8))
	plt.plot(t, t_loss, label="train", color="b")
	plt.plot(t, v_loss, label="valid", color="r")
	plt.legend(fontsize=18)
	plt.xlabel("Epoch", fontsize=18)
	plt.ylabel("Loss", fontsize=18)
	plt.title("Loss Curve", fontsize=18)
	plt.ylim(0.17, 0.40)
	plt.savefig(path)
	plt.clf()
