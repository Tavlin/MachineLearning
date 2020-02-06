import matplotlib.pyplot as plt

def jets():

    #particles = open("particles_default.txt", "r")
    particles = open("particles_y_up.txt", "r")
    #particles = open("particles_y_low.txt", "r")
    #jets = open("jets_default.txt", "r")
    jets = open("jets_y_up.txt", "r")
    #jets = open("jets_y_low.txt", "r")

    jets_origins = []
    jets_px = []
    jets_py = []
    jets_pz = []
    jets_color = []

    particles_origins = []
    particles_px = []
    particles_py = []
    particles_pz = []
    particles_color = []

    for line in jets:

        line = line.split(" ")

        if line[0] == '\n' or line[0] == '0': continue

        jets_origins.append(0)
        jets_px.append(float(line[0]))
        jets_py.append(float(line[1]))
        jets_pz.append(float(line[2]))

        jets_color.append('xkcd:olive')

    for line in particles:

        line = line.split(" ")

        if line[0] == '\n' or line[0] == '0': continue

        particles_origins.append(0)
        particles_px.append(float(line[0]))
        particles_py.append(float(line[1]))
        particles_pz.append(float(line[2]))

        particles_color.append('xkcd:navy blue')

    plt.subplots_adjust(left=None, bottom=.15, right=None, top=None, wspace=.4, hspace=.35)

    plt.subplot(121)

    plt.quiver(particles_origins, particles_origins, particles_px, particles_py, color=particles_color, angles='xy', scale_units='xy', scale=1)
    plt.quiver(jets_origins, jets_origins, jets_px, jets_py, color=jets_color, angles='xy', scale_units='xy', scale=1)

    plt.xlabel(r'p$_x$ (GeV)')
    plt.ylabel(r'p$_y$ (GeV)')

    plt.xlim(-10, 10)
    plt.ylim(-30, 30)
    plt.legend(['particles', 'jets'])

    plt.subplot(122)

    plt.quiver(particles_origins, particles_origins, particles_px, particles_pz, color=particles_color, angles='xy', scale_units='xy', scale=1)
    plt.quiver(jets_origins, jets_origins, jets_px, jets_pz, color=jets_color, angles='xy', scale_units='xy', scale=1)

    plt.xlabel(r'p$_x$ (GeV)')
    plt.ylabel(r'p$_z$ (GeV)')

    plt.xlim(-10, 10)
    plt.ylim(-40, 40)
    plt.legend(['particles', 'jets'])

    #plt.suptitle('Particle and Jet Momenta, rapidity 0.02')
    plt.suptitle('Particle and Jet Momenta, rapidity 0.2')
    #plt.suptitle('Particle and Jet Momenta, rapidity 0.0002')

    #plt.savefig("jets_default.png", dpi=300, format="png")
    plt.savefig("jets_rap_high.png", dpi=300, format="png")
    #plt.savefig("jets_rap_low.png", dpi=300, format="png")

    plt.show()


if __name__ == "__main__":
	jets()