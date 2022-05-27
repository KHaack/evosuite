"""
    Purpose: Run large scale experiments on a cluster.
    Author: Kevin Haack
"""
import argparse
import logging
import os
import sys
import json
import paramiko
from ping3 import ping
from scp import SCPClient

# paths and files
from experiment_lib import RunnerStatus

FILE_TEMP = "temp.log"
LOCATION_LOG_REMOTE = "/home/user/Benchmark/output.log"
LOCATION_STATUS_REMOTE = "/home/user/Benchmark/status.log"
LOCATION_LIB = "C:\\Users\\kha\\repos\\evosuite\\scripts\\experiment_lib.py"
LOCATION_LIB_REMOTE = "/home/user/Benchmark/experiment_lib.py"
LOCATION_SCRIPT = "C:\\Users\\kha\\repos\\evosuite\\scripts\\experiment_runner.py"
LOCATION_SCRIPT_REMOTE = "/home/user/Benchmark/experiment_runner.py"
LOCATION_JAR = "C:\\Users\\kha\\repos\\evosuite\\master\\target\\evosuite-master-1.2.1-SNAPSHOT.jar"
LOCATION_JAR_REMOTE = "/home/user/Benchmark/evosuite-master-1.2.1-SNAPSHOT.jar"
LOCATION_SAMPLE_REMOTE = "/home/user/Benchmark/samples/12 - new default - 713.txt"
LOCATION_CORPUS_REMOTE = "/home/user/Benchmark/SF110-20130704"
# the command that should be executed on the remotes
REMOTE_COMMAND = f'python3 "{LOCATION_SCRIPT_REMOTE}" -sample "{LOCATION_SAMPLE_REMOTE}" -corpus "{LOCATION_CORPUS_REMOTE}" -evosuite "{LOCATION_JAR_REMOTE}" -write_status -reboot'
# logging
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
LOG_LEVEL = logging.INFO
LOG_STREAM = sys.stdout
# remote computer
COPY_JAR = False
COPY_SCRIPT = True
COPY_LIB = True
ACCEPT_EVERY_SSH_KEY = True
CLUSTER_IPS = [
    '192.168.178.68',  # cluster0671
    '192.168.178.69',  # cluster0521
    '192.168.178.70',  # cluster0162
]
USERNAME = 'user'
PASSWORD = 'user'


def get_ssh(ip):
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


def monitor_remote(ip):
    """
    Monitor the remote with the passed ip
    :param ip: The ip of the remote.
    :return: None
    """
    with SCPClient(get_ssh(ip).get_transport()) as scp:
        logging.info(f'get status of {ip}...')
        scp.get(LOCATION_STATUS_REMOTE, FILE_TEMP)

        with open(FILE_TEMP, 'r') as f:
            content = json.loads(f.read())
            status = RunnerStatus(**content)
            status.print_status()
        os.remove(FILE_TEMP)


def start_remote(ip):
    """
    Run the experiment on the passed remote.
    :param ip: The remote ip.
    :return: None
    """
    ssh = get_ssh(ip)
    with SCPClient(ssh.get_transport()) as scp:
        if COPY_SCRIPT:
            logging.info(f'copy script on {ip}...')
            scp.put(LOCATION_SCRIPT, LOCATION_SCRIPT_REMOTE)

        if COPY_LIB:
            logging.info(f'copy lib on {ip}...')
            scp.put(LOCATION_LIB, LOCATION_LIB_REMOTE)

        if COPY_JAR:
            logging.info(f'copy jar on {ip}...')
            scp.put(LOCATION_JAR, LOCATION_JAR_REMOTE)

        logging.info(f'start script on {ip}...')
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(REMOTE_COMMAND)


def get_reachable_ips():
    """
    Ping the remotes and return the reachable ips.
    :return: The reachable ips.
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


def start_remotes():
    """
    Run the experiments on the remotes.
    :return: None
    """
    logging.info('start...')
    logging.info(f"remote command:\t{REMOTE_COMMAND}")

    for ip in get_reachable_ips():
        start_remote(ip)


def monitor_remotes():
    """
    Monitor the experiments on the remotes.
    :return: None
    """
    logging.info('monitor...')
    for ip in get_reachable_ips():
        monitor_remote(ip)


def setupArgparse():
    """
    Setup the argparse.
    :return: The parser
    """
    argument_parser = argparse.ArgumentParser(description="Run large scale experiments on a cluster.",
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = argument_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-start", help="Start the script on the remotes", action='store_true')
    group.add_argument("-monitor", help="Monitor the script on the remotes", action='store_true')
    group.add_argument("-ping", help="Ping the remotes", action='store_true')

    return argument_parser


def main():
    logging.basicConfig(stream=LOG_STREAM, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

    if args.monitor:
        monitor_remotes()
    elif args.start:
        start_remotes()
    elif args.ping:
        get_reachable_ips()


if __name__ == "__main__":
    args = setupArgparse().parse_args()
    main()
