'''
Perform a single run of the flow without control
'''
import os
import socket
import numpy as np
import csv

from tensorforce.agents import Agent
from tensorforce.execution import Runner

from env import resume_env, nb_actuations, simulation_duration

example_environment = resume_env(plot=False, dump_CL=100, dump_debug=1, dump_vtu=50, single_run=True)

deterministic = True

network = [dict(type='dense', size=512), dict(type='dense', size=512)]

if(os.path.exists("saved_models/test_strategy.csv")):
    os.remove("saved_models/test_strategy.csv")

if(os.path.exists("saved_models/test_strategy_avg.csv")):
    os.remove("saved_models/test_strategy_avg.csv")

def one_run():
    print("start simulation")
    state = example_environment.reset()
    example_environment.render = True
    null_action = np.zeros(example_environment.actions()['shape'])

    action_step_size = simulation_duration / nb_actuations  # Duration of 1 train episode / actions in 1 episode
    single_run_duration = 250  # In non-dimensional time
    action_steps = int(single_run_duration / action_step_size)

    for k in range(action_steps):
        state, terminal, reward = example_environment.execute(null_action)

    print("finish simulation\n")

    # Get avg quantities for the second half of the single run
    data = np.genfromtxt("saved_models/test_strategy.csv", delimiter=";")
    data = data[1:,1:]
    m_data = np.average(data[len(data)//2:], axis=0)
    nb_jets = len(m_data)-4
    # Print statistics
    print("Single Run finished. AvgDrag : {}, AvgRecircArea : {}".format(m_data[1], m_data[3]))

    name = "test_strategy_avg.csv"
    if(not os.path.exists("saved_models")):
        os.mkdir("saved_models")
    if(not os.path.exists("saved_models/"+name)):
        with open("saved_models/"+name, "w") as csv_file:
            spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
            spam_writer.writerow(["Name", "Drag", "Lift", "RecircArea"] + ["Jet" + str(v) for v in range(nb_jets)])
            spam_writer.writerow([example_environment.simu_name] + m_data[1:].tolist())
    else:
        with open("saved_models/"+name, "a") as csv_file:
            spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
            spam_writer.writerow([example_environment.simu_name] + m_data[1:].tolist())


if not deterministic:
    for _ in range(10):
        one_run()

else:
    one_run()
