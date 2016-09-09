import matplotlib.pyplot as plt
import matplotlib
import cPickle
import numpy as np

def bdt_old_ratio(data, strategy, baseline_strategy, lower_bound):

	matplotlib.rcParams.update({'font.size': 16})
	plt.hlines(1, lower_bound, 0.95, linestyles='dashed', linewidth=0.5)
	color = iter(plt.cm.rainbow(np.linspace(0, 1, len(data.keys()))))

	for ss in sorted(data.keys()): # loop thru the different signal samples
		c = next(color)
		if len(data[ss]['BDT'][1]) > 1:
			_ = plt.plot(data[ss]['BDT'][0], data[ss]['BDT'][1] / data[ss][baseline_strategy][1], label=ss, color=c, linewidth=2)
			maxidx = np.argmax((data[ss]['BDT'][1] / data[ss][baseline_strategy][1])[:-1])
			_ = plt.scatter(
				data[ss]['BDT'][0][maxidx], 
				(data[ss]['BDT'][1] / data[ss][baseline_strategy][1])[maxidx],
				color=c
				)
		else:
			raise ValueError("There are no data points to plot for class " + ss)

	plt.title('Asimov significance ratio wrt {} strategy'.format(baseline_strategy))
	plt.xlabel('BDT Threshold Value')
	plt.ylabel(r'$Z_{\mathrm{Asimov}}^{\mathrm{BDT}} / Z_{\mathrm{Asimov}}^{\mathrm{baseline}}$')
	plt.xlim(xmin=lower_bound, xmax=0.95)
	plt.ylim(ymin=0.2, ymax=2.8)

	plt.legend(loc='upper left')
	plt.savefig('threshold_ratio_{}.pdf'.format(strategy))
	plt.show()
	plt.clf()