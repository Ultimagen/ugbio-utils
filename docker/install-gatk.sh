#!/bin/bash
set -euo pipefail

# Install GATK into /opt/gatk (+ /usr/local/bin/gatk) for the ugbio member images.
#
# Uses the Ultima build of GATK 4.7.0.0 with netty-bom 4.1.139.Final (fixes CVE-2026-75595),
# published at https://github.com/Ultimagen/gatk/releases (BIOIN-3161). The official GATK
# releases still bundle a vulnerable netty. Switch back to an official release once one ships the fix.
#
# Requires wget and unzip.

GATK_VERSION="4.7.0.0-ug.netty1"
GATK_SHA256="044b98ad949d5a93eaf7666b5c2b5145741f68d7fd6a4fe75da109b83be02f78"
GATK_URL="https://github.com/Ultimagen/gatk/releases/download/${GATK_VERSION}/gatk-${GATK_VERSION}.zip"

echo "Installing GATK ${GATK_VERSION} ..."
wget -q -O /tmp/gatk.zip "${GATK_URL}"
echo "${GATK_SHA256}  /tmp/gatk.zip" | sha256sum -c -
unzip -q /tmp/gatk.zip -d /opt
mv "/opt/gatk-${GATK_VERSION}" /opt/gatk
# The spark jar is only needed for --spark-runner SPARK/GCS; Spark tools in local mode use the local jar
rm -f /opt/gatk/gatk-package-*-spark.jar
ln -sf /opt/gatk/gatk /usr/local/bin/gatk
rm -f /tmp/gatk.zip

echo "Done. GATK ${GATK_VERSION} installed in /opt/gatk"
