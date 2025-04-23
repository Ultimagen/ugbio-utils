#!/bin/bash

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
        -v|--version)    shift
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

OUTPUT_PATH=""
HAIL_VERSION="master"
SPARK_VERSION="2.4.0"
COMPILE=true
IS_MASTER=false
GRADLE_DEPRECATION=1566593776
SELECTED_VERSION=$GRADLE_DEPRECATION
export TEST=""
export CXXFLAGS=-march=native

if grep isMaster /mnt/var/lib/info/instance.json | grep true;
then
  IS_MASTER=true
fi

while [ $# -gt 0 ]; do
    case "$1" in
    --output-path)
      shift
      OUTPUT_PATH=$1
      ;;
    --hail-version)
      shift
      HAIL_VERSION=$1
      ;;
    --spark-version)
      shift
      SPARK_VERSION=$1
      ;;
    -*)
      error_msg "unrecognized option: $1"
      ;;
    *)
      break;
      ;;
    esac
    shift
done

echo "Building Hail from $HASH"

if [ "$IS_MASTER" = true ]; then
    # Install development tools and dependencies
    sudo yum groupinstall "Development Tools" -y
#    sudo yum install git cmake lz4 lz4-devel python3 -y
#    python3 -m ensurepip --upgrade
#    sudo yum install java-1.8.0-amazon-corretto-devel -y
    sudo amazon-linux-extras enable corretto11
    sudo yum install -y git htop unzip bzip2 zip tar rsync emacs-nox xsltproc java-11-amazon-corretto-devel cmake gcc gcc-c++ lapack-devel lz4-devel


    export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
    export PATH=$JAVA_HOME/bin:$PATH
    # Clone Hail repository
    git clone https://github.com/broadinstitute/hail.git
    cd hail/hail/
    git checkout $HAIL_VERSION
    GIT_HASH="$(git log --pretty=format:"%H" | grep $HASH | cut -f 1 -d ' ')"

    if [ ${#HASH} -lt 7 ]; then
    	if [ $HASH = "current" ]; then
    		echo "Hail will be compiled using the latest repository version available"
    	else
    		echo "The git hash provided has less than 7 characters. The latest version of Hail will be compiled!"
    		# exit 1
    	fi
    else
    	export TEST="$(aws s3 ls s3://hms-dbmi-docs/hail-versions/ | grep $HASH | sed -e 's/^[ \t]*//' | cut -d " " -f 2)"
    	if [ -z "$TEST" ] || [-z "$GIT_HASH" ]; then
    		echo "Hail pre-compiled version not found!"
            echo "Compiling Hail with git hash: $GIT_HASH"
            git reset --hard $GIT_HASH
            SELECTED_VERSION=`git show -s --format=%ct $GIT_HASH`
    	else
    		echo "Hail pre-compiled version found: $TEST"
            aws s3 cp s3://hms-dbmi-docs/hail-versions/$TEST $HOME/ --recursive
            GIT_HASH="$(echo $TEST | cut -d "-" -f 1)"
            git reset --hard $GIT_HASH
            COMPILE=false
    	fi
    fi

    LATEST_JDK=`ls  /usr/lib/jvm/ | grep "jre-11-openjdk"`
    sudo  ln -s /usr/lib/jvm/$LATEST_JDK/include /etc/alternatives/jre/include


    if [ "$COMPILE" = true ]; then
        # Compile with Spark 2.4.0
        if [ $SELECTED_VERSION -ge $GRADLE_DEPRECATION ];then
          echo "Compiling with Wheel..."
          make clean
          make wheel
          HAIL_WHEEL=`ls /opt/ugbio-utils/src/hail/hail/hail/build/deploy/dist | grep "whl"`
          sudo python3 -m pip install --no-deps /opt/ugbio-utils/src/hail/hail/hail/build/deploy/dist/$HAIL_WHEEL

      else  ./gradlew -Dspark.version=$SPARK_VERSION -Dbreeze.version=0.13.2 -Dpy4j.version=z0.10.6 shadowJar archiveZip
            cp $PWD/build/distributions/hail-python.zip $HOME
            cp $PWD/build/libs/hail-all-spark.jar $HOME
        fi
    fi
fi
