from multiprocessing import Process
import time
import argparse
import socket
import os
from utils import check_ports_avail

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
ap.add_argument("-p", "--ports-start", required=True, help="the start of the range of ports to use", type=int)
ap.add_argument("-t", "--host", default="None", help="the host; default is local host; string either internet domain or IPv4", type=str)

args = vars(ap.parse_args())

number_servers = args["number_servers"]
ports_start = args["ports_start"]
host = args["host"]

# update the mesh stuff with the latest version
os.system("cp -r simulation_base/mesh .")

# If no host specified, use local
if host == 'None':
    host = socket.gethostname()

# create list of ports to be used
list_ports = [ind_server + ports_start for ind_server in range(number_servers)]

# check for the availability of the ports, if not available, quit
if not check_ports_avail(host, list_ports):
    quit()

# make copies of the code to avoid collisions  --> env_{environment rank}
for rank in range(number_servers):

    print("copy simulation_base for env of rank {}".format(rank))

    # make the env folder and copy all the necessary files
    if not os.path.exists('env_' + str(rank)):
        os.system('mkdir ' + 'env_' + str(rank) + '/')

    # copy simulation_base to each env folder
    os.system('cp -r simulation_base/* env_' + str(rank) + '/.')

    # create text file called rank.txt containing env rank inside each env folder
    os.system('touch env_{}/rank'.format(rank))
    os.system('echo {} >> env_{}/rank'.format(rank, rank))

def launch_one_server(rank, host, port):
    '''
    Go into env_rank folder and start server for that environment
    '''
    os.system('cd env_{} && python3 start_one_server.py -t {} -p {}'.format(rank, host, port))	        

processes = []

# launch all the servers one after the other
for rank, port in enumerate(list_ports):
    print("launching process of rank {}".format(rank))
    proc = Process(target=launch_one_server, args=(rank, 'localhost', port))  # Spawn child launch_one_server process
    proc.start()  # start the process
    processes.append(proc)
    time.sleep(2.0)  # just to avoid collisions in the terminal printing

print("all processes started, ready to serve...")

for proc in processes:
    proc.join()  # Ensure that the main process waits for each child process to complete
