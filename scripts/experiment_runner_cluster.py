"""
    Purpose: Run large scale experiments on a cluster.
    Author: Kevin Haack
"""
import argparse
import datetime
import json
import logging
import os
from datetime import timedelta, datetime

import paramiko
from ping3 import ping
from scp import SCPClient

import experiment_lib as ex
# paths and files
from experiment_lib import ExperimentRunner, Status

FILE_TEMP = "temp.log"
LOCATION_LOG_REMOTE = "/home/user/Benchmark/output.log"
LOCATION_STATUS_REMOTE = "/home/user/Benchmark/status.log"
LOCATION_LIB = "C:\\Users\\kha\\repos\\evosuite\\scripts\\experiment_lib.py"
LOCATION_LIB_REMOTE = "/home/user/Benchmark/experiment_lib.py"
LOCATION_SCRIPT = "C:\\Users\\kha\\repos\\evosuite\\scripts\\experiment_runner.py"
LOCATION_SCRIPT_REMOTE = "/home/user/Benchmark/experiment_runner.py"
LOCATION_JAR = "C:\\Users\\kha\\repos\\evosuite\\master\\target\\evosuite-master-1.2.1-SNAPSHOT.jar"
LOCATION_JAR_REMOTE = "/home/user/Benchmark/evosuite-master-1.2.1-SNAPSHOT.jar"
LOCATION_SAMPLE = "C:\\Users\\kha\\Desktop\\Benchmark\\samples\\15 - kifetew selected classes - 346.txt"
LOCATION_SAMPLE_REMOTE = "/home/user/Benchmark/samples/15 - kifetew selected classes - 346.txt"
LOCATION_CORPUS_REMOTE = "/home/user/Benchmark/SF110-20130704"
# the command that should be executed on the remotes
REMOTE_COMMAND = f'python3 "{LOCATION_SCRIPT_REMOTE}" -sample "{LOCATION_SAMPLE_REMOTE}" -corpus "{LOCATION_CORPUS_REMOTE}" -evosuite "{LOCATION_JAR_REMOTE}" -write_status -reboot'
# remote computer
COPY_JAR = True
COPY_SCRIPT = True
COPY_LIB = True
COPY_SAMPLE = True
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
    :return: ExperimentRunner of the passed ip.
    """
    with SCPClient(get_ssh(ip).get_transport()) as scp:
        logging.info(f'get status of {ip}...')
        scp.get(LOCATION_STATUS_REMOTE, FILE_TEMP)

        with open(FILE_TEMP, 'r') as f:
            content = json.loads(f.read())
            runner = ExperimentRunner(**content)
        os.remove(FILE_TEMP)

    return runner


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

        if COPY_SAMPLE:
            logging.info(f'copy sample on {ip}...')
            scp.put(LOCATION_SAMPLE, LOCATION_SAMPLE_REMOTE)

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


def create_cluster_report(runners):
    """
    Creates a cluster report.
    :param runners: A list of all runners.
    :return: None
    """
    logging.info('create report...')

    executions_total = 0
    executions_done = 0
    sample_total = 0
    sample_done = 0
    max_runtime = 0

    logging.info('---------------------------------------------------------------------------------------')
    for runner in runners:
        runner.print_status()

        if runner.status == Status.RUNNING:
            sample_total = sample_total + runner.sample_size
            sample_done = sample_done + runner.current_class_index
            executions_total = executions_total + (runner.sample_size * runner.executions_per_class)
            executions_done = executions_done + (runner.current_class_index * runner.executions_per_class) + runner.current_execution

            runtime = runner.get_runtime_estimation()

            if runtime > max_runtime:
                max_runtime = runtime

        logging.info('---------------------------------------------------------------------------------------')

    delta = timedelta(seconds=max_runtime)
    end = datetime.now() + delta

    logging.info(f'Estimated runtime {str(max_runtime / 60 / 60)}h')
    logging.info(f'Estimated end {end.strftime("%Y-%m-%d %H-%M-%S")}h')
    logging.info(f'Class {str(sample_done)}/{str(sample_total)}')
    logging.info(f'Executions {str(executions_done)}/{str(executions_total)}')


def monitor_remotes():
    """
    Monitor the experiments on the remotes.
    :return: None
    """
    logging.info('monitor...')

    runners = []
    for ip in get_reachable_ips():
        runner = monitor_remote(ip)
        runners.append(runner)

    create_cluster_report(runners)


def setup_argparse():
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
    if args.monitor:
        monitor_remotes()
    elif args.start:
        start_remotes()
    elif args.ping:
        get_reachable_ips()


if __name__ == "__main__":
    ex.init_default_logging()
    args = setup_argparse().parse_args()
    main()
