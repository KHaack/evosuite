"""
    Purpose: Run large scale experiments on a cluster.
    Author: Kevin Haack
"""
import paramiko
import logging
import sys
from scp import SCPClient
from ping3 import ping

# paths and files
SCRIPT_LOCATION = "C:\\Users\\kha\\repos\\evosuite\\scripts\\experiment_runner.py"
SCRIPT_LOCATION_REMOTE = "/home/user/Benchmark/experiment_runner.py"
REMOTE_COMMAND = f'python3 {SCRIPT_LOCATION_REMOTE}'
# logging
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
LOG_LEVEL = logging.INFO
LOG_STREAM = sys.stdout
# remote computer
ACCEPT_EVERY_SSH_KEY = True
CLUSTER_IPS = [
    '192.168.178.68',   # cluster0671
    '192.168.178.69',  # cluster0521
    '192.168.178.70',   # cluster0162
    ]
USERNAME = 'user'
PASSWORD = 'user'


def runRemote(ip):
    """
    Run the experiment on the passed remote.
    :param ip: The remote ip.
    :return: None
    """
    logging.info(f'connect to {ip}...')
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()

    if ACCEPT_EVERY_SSH_KEY:
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh.connect(hostname=ip,
                username=USERNAME,
                password=PASSWORD)

    with SCPClient(ssh.get_transport()) as scp:
        scp.put(SCRIPT_LOCATION, SCRIPT_LOCATION_REMOTE)
        logging.info(f'start script on {ip}...')
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(REMOTE_COMMAND)


def getReachable():
    """
    Ping the remotes.
    :return: The reachable machines.
    """
    logging.info('ping cluster...')

    reachable = []
    for ip in CLUSTER_IPS:
        if ping(ip):
            logging.info(f"{ip} reachable")
            reachable.append(ip)
        else:
            logging.info(f"{ip} failed")
    logging.info(f'{str(len(reachable))}/{str(len(CLUSTER_IPS))} reachable')

    return reachable

def runRemotes():
    """
    Run the experiments of the remotes.
    :return: None
    """
    for ip in getReachable():
        runRemote(ip)


def main():
    logging.basicConfig(stream=LOG_STREAM, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)
    logging.info('start...')
    runRemotes()


if __name__ == "__main__":
    main()
