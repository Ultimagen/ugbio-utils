if grep isMaster /mnt/var/lib/info/instance.json | grep true;
then
  IS_MASTER=true
fi

if [ "$IS_MASTER" = true ]; then
    # Install development tools and dependencies
    sudo yum groupinstall "Development Tools" -y
    sudo yum install -y git htop unzip bzip2 zip tar rsync emacs-nox xsltproc java-11-amazon-corretto-devel cmake gcc gcc-c++ lapack-devel lz4-devel


    export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
    export PATH=$JAVA_HOME/bin:$PATH

    sudo rm /etc/alternatives/jre/include/include
    LATEST_JDK=`ls  /usr/lib/jvm/ | grep "jre-11-openjdk"`
    sudo  ln -s /usr/lib/jvm/$LATEST_JDK/include /etc/alternatives/jre/include

    pip install pyspark hail
fi