import os
import socket
import numpy as np
import csv

from tensorforce.agents import Agent
from tensorforce.execution import Runner

from simulation_base.env import resume_env, nb_actuations, simulation_duration

example_environment = resume_env(plot=False, single_run=True, dump_debug=1)

deterministic = True

saver_restore = os.getcwd() + "/saver_data/"

agent = Agent.load(directory = saver_restore)

# If previous evaluation results exist, delete them
if(os.path.exists("saved_models/test_strategy.csv")):
    os.remove("saved_models/test_strategy.csv")

if(os.path.exists("saved_models/test_strategy_avg.csv")):
    os.remove("saved_models/test_strategy_avg.csv")

def one_run():
    print("Start simulation")
    state = example_environment.reset()
    example_environment.render = True

    action_step_size = simulation_duration / nb_actuations  # Duration of 1 train episode / actions in 1 episode
    single_run_duration = 250  # In non-dimensional time
    action_steps = int(single_run_duration / action_step_size)

    internals = agent.initial_internals()

    for k in range(action_steps):
        action, internals = agent.act(state, deterministic=deterministic, independent=True, internals=internals)
        state, terminal, reward = example_environment.execute(action)

    data = np.genfromtxt("saved_models/test_strategy.csv", delimiter=";")
    data = data[1:,1:]
    m_data = np.average(data[len(data)//2:], axis=0)  # Calculate means for the second half of the single episode
    nb_jets = len(m_data)-4
    # Print statistics
    print("Single Run finished. AvgDrag : {}, AvgRecircArea : {}".format(m_data[1], m_data[2]))

    # Output average values for the single run (Note that values for each timestep are already reported as we execute)
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
