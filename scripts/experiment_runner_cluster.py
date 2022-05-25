"""
    Purpose: Run large scale experiments on a cluster.
    Author: Kevin Haack
"""
import argparse
import logging
import os
import sys

import paramiko
from ping3 import ping
from scp import SCPClient

# paths and files
FILE_TEMP = "output.log"
LOCATION_LOG_REMOTE = "/home/user/Benchmark/output.log"
LOCATION_SCRIPT = "C:\\Users\\kha\\repos\\evosuite\\scripts\\experiment_runner.py"
LOCATION_SCRIPT_REMOTE = "/home/user/Benchmark/experiment_runner.py"
LOCATION_JAR = "C:\\Users\\kha\\repos\\evosuite\\master\\target\\evosuite-master-1.2.1-SNAPSHOT.jar"
LOCATION_JAR_REMOTE = "/home/user/Benchmark/evosuite-master-1.2.1-SNAPSHOT.jar"
LOCATION_SAMPLE_REMOTE = "/home/user/Benchmark/samples/10 - new default - 356.txt"
LOCATION_CORPUS_REMOTE = "/home/user/Benchmark/SF110-20130704"
# the command that should be executed on the remotes
REMOTE_COMMAND = f'python3 "{LOCATION_SCRIPT_REMOTE}" -sample "{LOCATION_SAMPLE_REMOTE}" -corpus "{LOCATION_CORPUS_REMOTE}" -evosuite {LOCATION_JAR_REMOTE}'
# logging
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
LOG_LEVEL = logging.INFO
LOG_STREAM = sys.stdout
# remote computer
COPY_JAR = False
COPY_SCRIPT = True
ACCEPT_EVERY_SSH_KEY = True
CLUSTER_IPS = [
    '192.168.178.68',  # cluster0671
    '192.168.178.69',  # cluster0521
    '192.168.178.70',  # cluster0162
]
USERNAME = 'user'
PASSWORD = 'user'


def getSSH(ip):
    """
    Return the ssh object for the passed ip.
    :param ip: The ip for the ssh connection.
    :return: The ssh object
    """
    logging.info(f'connect to {ip}...')
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()

    if ACCEPT_EVERY_SSH_KEY:
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh.connect(hostname=ip,
                username=USERNAME,
                password=PASSWORD)
    return ssh


def monitorRemote(ip):
    """
    Monitor the remote with the passed ip
    :param ip: The ip of the remote.
    :return: None
    """
    with SCPClient(getSSH(ip).get_transport()) as scp:
        logging.info(f'get log of {ip}...')
        scp.get(LOCATION_LOG_REMOTE)

        with open(FILE_TEMP, 'r') as f:
            last_line = f.readlines()[-1].replace('\n', '')
            logging.info(last_line)
        os.remove(FILE_TEMP)


def startRemote(ip):
    """
    Run the experiment on the passed remote.
    :param ip: The remote ip.
    :return: None
    """
    ssh = getSSH(ip)
    with SCPClient(ssh.get_transport()) as scp:
        if COPY_JAR:
            logging.info(f'copy script on {ip}...')
            scp.put(LOCATION_SCRIPT, LOCATION_SCRIPT_REMOTE)

        if COPY_JAR:
            logging.info(f'copy jar on {ip}...')
            scp.put(LOCATION_JAR, LOCATION_JAR_REMOTE)

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


def startRemotes():
    """
    Run the experiments on the remotes.
    :return: None
    """
    logging.info('start...')
    for ip in getReachable():
        startRemote(ip)


def monitorRemotes():
    """
    Monitor the experiments on the remotes.
    :return: None
    """
    logging.info('monitor...')
    for ip in getReachable():
        monitorRemote(ip)


def setupArgparse():
    """
    Setup the argparse.
    :return: The parser
    """
    parser = argparse.ArgumentParser(description="Run large scale experiments on a cluster.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-start", help="Start the script on the remotes", action='store_true')
    group.add_argument("-monitor", help="Monitor the script on the remotes", action='store_true')

    return parser


def main():
    logging.basicConfig(stream=LOG_STREAM, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

    if args.monitor:
        monitorRemotes()
    elif args.start:
        startRemotes()


if __name__ == "__main__":
    parser = setupArgparse()
    args = parser.parse_args()
    main()
