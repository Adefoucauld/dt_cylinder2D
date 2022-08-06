import argparse
import socket

import sys
import os

cwd = os.getcwd()
sys.path.append(cwd + "/../")

from RemoteEnvironmentServer import RemoteEnvironmentServer
from env import resume_env

ap = argparse.ArgumentParser()

ap.add_argument("-p", "--port", required=True, help="the port to use", type=int)
ap.add_argument("-t", "--host", default="None", help="the host; default is local host; string either internet domain or IPv4", type=str)

args = vars(ap.parse_args())

port = args["port"]
host = args["host"]

# We are within one of the env folders. read rank.txt to know which one
with open('rank', 'r') as fh:
    rank_line = fh.readline()
    rank = int(rank_line)  # get rank number
    print("This is the simulation of rank {}".format(rank))

# If no host specified, use local
if host == 'None':
    host = socket.gethostname()

def check_free_port(host, port, verbose=True):
    """Check if a given port is available."""
    sock = socket.socket()
    try:
        sock.bind((host, port))
        sock.close()
        print("host {} on port {} is AVAIL".format(host, port))
        return(True)
    except:
        print("host {} on port {} is BUSY".format(host, port))
        sock.close()
        return(False)


if not check_free_port(host, port):
    print("the port is not available; quitting!")
    quit()


def launch_server(host, port):
    # to avoid cluttering the terminal...
    if rank == 0:
        print("launch with a lot of output, this is rank 0")
        tensorforce_environment = resume_env(plot=False, dump_debug=1)
        RemoteEnvironmentServer(tensorforce_environment, host=host, port=port)
        
    else:
        print("Launch with less output, this is higher rank")
        tensorforce_environment = resume_env(plot=False, dump_debug=1, dump_CL=False)  # No CL output
        RemoteEnvironmentServer(tensorforce_environment, host=host, port=port, verbose=0)


launch_server(host, port)
