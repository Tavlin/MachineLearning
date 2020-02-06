import matplotlib.pyplot as plt

def rapidity():

	muon_x_axis = []
	total_x_axis = []

	muon_rapidity = []
	muon_error_plus = []
	muon_error_minus = []

	total_rapidity = []
	total_error_plus = []
	total_error_minus = []

	scale = 100

	muon = open("muon.txt", "r")
	total = open("total.txt", "r")

	for line in muon:
		try:
			line=list(map(float,line.split(" ")))
		except:
			print("ignore comment line")
		else:
			muon_error_minus.append(line[1]-line[0])
			muon_error_plus.append(line[2]-line[1])
			muon_x_axis.append(line[1])
			muon_rapidity.append(line[3]*scale)
		
	for line in total:
		try:
			line=list(map(float,line.split(" ")))
		except:
			print("ignore comment line")
		else:
			total_error_minus.append(line[1] - line[0])
			total_error_plus.append(line[2] - line[1])
			total_x_axis.append(line[1])
			total_rapidity.append(line[3])

	muon.close()
	total.close()

	# -- Plotting --

	plt.errorbar(total_x_axis, total_rapidity, xerr=[total_error_minus, total_error_minus], fmt='d', color='xkcd:maroon')
	plt.errorbar(muon_x_axis, muon_rapidity, xerr=[muon_error_minus, muon_error_minus], fmt='d', color='xkcd:dark pink', fillstyle='none')
	plt.xlabel('rapidity')
	plt.ylabel(r'count / N$_{ev}$')
	#plt.yscale('log')

	plt.legend(['all particles', 'muons x100'])
	plt.title('Rapidity Distribution, 10.000 Events')

	plt.savefig("rapidity.png", dpi = 300, format="png")


	plt.show()


if __name__ == "__main__":
	rapidity()
