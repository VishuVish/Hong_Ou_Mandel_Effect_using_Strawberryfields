#Hanbury_Brown_Twiss_(HBT)_Experiment

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit.circuit.library import UnitaryGate
import numpy as np

#Experimental_Setup
qr = QuantumRegister(2, name="q") #2 photon modes 
cr = ClassicalRegister(2, name="c") #measurement bits for detection event
qc = QuantumCircuit(qr,cr) #You can think of this as our optical bench equivalent for qiskit 
qc.x(qr[0])#Initializing the qubit according to the experimental setup

#Beam_Splitter
t = 1/np.sqrt(2)
r = 1j/np.sqrt(2)
bs_4x4 = np.array([[1,0,0,0],[0,t,r,0],[0,r,t,0],[0,0,0,1]]) #Unitary Matrix of Beam Splitter that acts on the input state
bs_gate = UnitaryGate(bs_4x4, label="BS") #Now this is Beam Splitter is a Quantum Gate that could act of "qr" bits 

#Singe Photon Source
#Simulating the Experiment
qc.append(bs_gate, [qr[0],qr[1]]) #Now the Beam Splitter Gate of the Quantum Circuit acts on the input qr bits
qc.measure (qr, cr) #This measures the qr bit and registers the measured qr bit as cr bit (detection event)

# Use Aer simulator backend
sim_01 = AerSimulator()
qc_t = transpile(qc, sim_01)

# Execute the circuit with n shots, n equal or greater than 1000
job = sim_01.run(qc_t, shots=10000)
result = job.result()
counts = result.get_counts()

print("Measurement results for Single Photon Source:", counts)

#plotting the results
x = list(counts.keys())
y = list(counts.values())

labels = {'10': r'$|10\rangle$','01': r'$|01\rangle$','11': r'$|11\rangle$','00': r'$|00\rangle$'}

Single_Photon_counts = {labels.get(k, k): v for k, v in counts.items()}

plot_histogram(Single_Photon_counts)
plt.xlabel("Measurement Outcome")
plt.ylabel("Counts")
plt.title("Plot for Single Photon Source")
plt.show()

#Second Order Correlation Function g(2)(0) Calculation 
N_01_SPS = counts.get('01', 0) #Counts for the state |01>
N_10_SPS = counts.get('10', 0) #Counts for the state |10>
N_11 = counts.get('11',0) #Counts for the state |11> (No coincidence event occurs for Single Photon Source)

g_2_SPS = N_11 / (N_01_SPS * N_10_SPS) if (N_01_SPS + N_10_SPS) > 0 else print("Cannot calculate g(2)")
print("The Second order Correlation function for Single Photon Source (Antibunching): ",g_2_SPS)


#Thermal_Source

from qiskit.quantum_info import DensityMatrix, Operator, Statevector

n_avg = 1.5 #Obtain this value from Bose Einstein Condensate
p1 = n_avg/(1+n_avg)

simulator = AerSimulator()
counts_for_thermal = {}

#Measurement Simulation for Thermal Source
for _ in range (10000):
    qr = QuantumRegister(2, "qr")
    cr = ClassicalRegister(2, "cr")
    qc = QuantumCircuit(qr,cr)
    
    """
    The following 4 lines of the code are basically like random number generators, 
    so for the predefined p1, which shows the probability distribution of the light in Thermal source, 
    we are comparing it to the generated photon state.
    If the generated Photon State is having a probability distribution greater than p1, 
    we discard it as it would mean that the generated photon has more intensity than what we incident onto the beam splitter. 
    If the generated photon state is less than p1, in whichever the mode such photon appear, 
    the particular mode gets flipped from 0 to 1, indicating the presence of photon in the respective mode.
    The result overshoots i.e., greater than 2 for thermal source, which is lot higher. This is because the code doesn't 
    account for correlation between the two modes, which is a characteristic of thermal light sources.
    """
    if np.random.rand() < p1: 
        qc.x(qr[0])
    if np.random.rand() < p1: 
        qc.x(qr[1])
       
    #Applying Quantum Circuit  
    qc.append(bs_gate, [[qr[0]],qr[1]])
    qc.measure(qr,cr)
    
    #Simulating the Quantum Circuit in Aer Simulator
    result = simulator.run(qc, shots=1).result()
    counts = result.get_counts()

    for k in counts:
        counts_for_thermal[k] = counts_for_thermal.get(k, 0) + counts[k]
        
print("Measurement results for Thermal Photon Source:", counts_for_thermal)

#Plotting the results
thermal_counts = {labels.get(k, k): v for k, v in counts_for_thermal.items()}

plot_histogram(thermal_counts)
plt.xlabel("Measurement Outcome")
plt.ylabel("Counts")
plt.title("Plot for Thermal Photon Source")
plt.show()

#Second order Correlation Function g(2)(0) Calculation for Thermal Source
N_00_TPS = counts_for_thermal.get('00', 0) 
N_01_TPS = counts_for_thermal.get('01', 0) 
N_10_TPS = counts_for_thermal.get('10', 0)
N_11_TPS = counts_for_thermal.get('11', 0) 
N_total_TPS = sum(counts_for_thermal.values())

g_2_TPS = (N_11_TPS*N_total_TPS) / (N_01_TPS * N_10_TPS) if (N_01_TPS + N_10_TPS) > 0 else print("Cannot calculate g(2)")
print("The Second order Correlation function for Thermal Photon Source (Bunching): ", g_2_TPS)

#Coherent Light Source

coherent_Amp = 2.0
coherent_Circuit = QuantumCircuit(2, 2) #2 photon modes 

""" In a Coherent Light Source, the amplitude of the coherent state is represented by coherent_Amp.
    Here, we are using a rotation gate to simulate the coherent state preparation.
    Basically, making both the qubits in the modes 0 and 1 are having same amplitude and phase, mimicking LASER light
"""

coherent_Circuit.rx(np.pi/4, 0) 
coherent_Circuit.rx(np.pi/4, 1)

#Coherent Light Source incident to Beam Splitter
coherent_Circuit.append(bs_gate, [0, 1]) 
coherent_Circuit.measure([0,1],[0,1]) 

#Measurement Simulation
simulator = AerSimulator()
res = simulator.run(coherent_Circuit, shots=10000).result()
counts_coherent = res.get_counts()

counts_for_coherent = {}
for k in counts_coherent:
    counts_for_coherent[k] = counts_for_coherent.get(k, 0) + counts_coherent[k]
print("Measurement results for Coherent Photon Source:", counts_for_coherent)

coherent_counts = {labels.get(k, k): v for k, v in counts_for_coherent.items()}

plot_histogram(coherent_counts) 
plt.xlabel("Measurement Outcome")
plt.ylabel("Counts")
plt.title("Plot for Coherent Photon Source")
plt.show()

#Second order Correlation Function g(2)(0) Calculation for Coherent Source
N_00_CPS = counts_for_coherent.get('00', 0)
N_01_CPS = counts_for_coherent.get('01', 0) 
N_10_CPS = counts_for_coherent.get('10', 0)
N_11_CPS = counts_for_coherent.get('11', 0)
N_total_CPS = sum(counts_for_coherent.values())

g_2_CPS = (N_11_CPS*N_total_CPS)/(N_01_CPS*N_10_CPS) if (N_01_CPS + N_10_CPS) > 0 else print("Cannot calculate g(2)")
print("The Second order Correlation function for Coherent Photon Source: ", g_2_CPS)
