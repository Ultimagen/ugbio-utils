#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>/tmp/cloudcreation_log.out 2>&1

export HAIL_HOME="/opt/ugbio-utils"
export HASH="current"

# Error message
error_msg ()
{
  echo 1>&2 "Error: $1"
  exit 1
}

# Usage
usage()
{
echo "Usage: cloudformation.sh [-v | --version <git hash>] [-h | --help]

Options:
-v | --version <git hash>
    This option takes either the abbreviated (8-12 characters) or the full size hash (40 characters).
    When provided, the command uses a pre-compiled Hail version for the EMR cluster. If the hash (sha1)
    version exists in the pre-compiled list, that specific hash will be used.
    If no version is given or if the hash was not found, Hail will be compiled from scratch using the most
    up to date version available in the repository (https://github.com/hail-is/hail)

-h | --help
	Displays this menu"
    exit 1
}

# Read input parameters
while [ "$1" != "" ]; do
    case $1 in
        -v|--version)	shift
                        HASH="$1"
                        ;;
        -h|--help)      usage
                        ;;
        -*)
      					error_msg "unrecognized option: $1"
      					;;
        *)              usage
    esac
    shift
done

chmod 700 $HOME/.ssh/id_rsa/
KEY=$(ls ~/.ssh/id_rsa/)

for WORKERIP in `sudo grep -i privateip /mnt/var/lib/info/*.txt | sort -u | cut -d "\"" -f 2`
do
   # Distribute keys to workers
   scp -o "StrictHostKeyChecking no" -i ~/.ssh/id_rsa/${KEY} ~/.ssh/authorized_keys ${WORKERIP}:/home/hadoop/.ssh/authorized_keys
done

echo 'Keys successfully copied to the worker nodes'

# Add hail to the master node
sudo mkdir -p /opt
sudo chmod 777 /opt/
sudo chown hadoop:hadoop /opt
cd /opt
git clone --branch hail_install https://github.com/Ultimagen/ugbio-utils.git
cd $HAIL_HOME/src/hail

# Compile Hail
./hail_install.sh

# Set the time zone for cron updates
sudo cp /usr/share/zoneinfo/America/New_York /etc/localtime

# Get IPs and names of EC2 instances (workers) to monitor if a worker dropped  
sudo grep -i privateip /mnt/var/lib/info/*.txt | sort -u | cut -d "\"" -f 2 > /tmp/t1.txt
CLUSTERID="$(jq -r .jobFlowId /mnt/var/lib/info/job-flow.json)"
aws emr list-instances --cluster-id ${CLUSTERID} | jq -r .Instances[].Ec2InstanceId > /tmp/ec2list1.txt

# Setup crontab to check dropped instances every minute and install SW as needed in new instances 
sudo echo "* * * * * /opt/hail-on-AWS-spot-instances/src/run_when_new_instance_added.sh >> /tmp/cloudcreation_log.out 2>&1 # min hr dom month dow" | crontab -

./jupyter_run.sh

