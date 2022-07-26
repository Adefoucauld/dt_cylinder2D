import os
import socket
import numpy as np
import csv
from collections import deque

from tensorforce.agents import Agent

saver_restore = os.getcwd() + "/saver_data/"

agent = Agent.load(directory = saver_restore)

# If folder does not exist, create it
if(not os.path.exists("frequency_response")):
        os.mkdir("frequency_response")

### System parameters ###

# Vortex shedding cycle
t_vs = 6.860
# Forcing sampling time
t_s = 1.0/100.0
# Action time of controller
t_a = 0.5

### Analysis parameters ###

length = 100
start_freq = 1/length
stop_freq = 10.0*(1/t_vs)
num_freqs = 50
amplitude = 0.01

### Controller harmonic forcing

def one_run(frequency=1, length = 10*t_vs, t_s = t_s):
    
    omega = 2*np.pi*frequency

    internals = agent.initial_internals()
    ANN_IO = []

    # Get information about ANN inputs
    num_history_steps = len(agent.states_spec)

    for k in range(int(length/t_s)):
        # Update current state
        state = {'obs': np.array(amplitude*np.sin(omega*k*t_s)).reshape((1,))}
        # Update delayed states to mimic past_observations
        for past_obs in range(num_history_steps-1):
            key = "prev_obs_" + str(past_obs + 1)
            t_prev = k*t_s - (past_obs + 1)*t_a
            state.update({key : np.array(amplitude*np.sin(omega*t_prev)).reshape((1,))})

        action, internals = agent.act(state, evaluation=True, internals=internals)
        ANN_IO.append([k*t_s,state['obs'][0],action[0]])

    # Output average values for the single run (Note that values for each timestep are already reported as we execute)
    name = "IO_f_" + "{:.5f}".format(frequency).replace('.','p') + ".csv"
    
    with open("frequency_response/"+name, "w") as csv_file:
        spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
        spam_writer.writerow(["Time", "Input", "Output"])
        spam_writer.writerows(ANN_IO)


print("Starting frequency response analysis: start-frequency = {:.3f} - stop-frequency = {:.3f}".format(start_freq,stop_freq))

freqs = np.logspace(start=np.log10(start_freq), stop=np.log10(stop_freq), num=num_freqs)

for freq in freqs:
    print("frequency = {:.3f}".format(freq))
    one_run(frequency=freq, length=length)

print("Analysis complete - Exiting")
