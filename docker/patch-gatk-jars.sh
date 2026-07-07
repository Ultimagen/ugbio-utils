#!/bin/bash
set -euo pipefail

# Patch GATK fat JAR to fix bundled dependency CVEs:
#   - CVE-2026-42581, CVE-2026-42584: netty 4.1.118 -> 4.1.135
#   - CVE-2024-51504: zookeeper 3.9.2 -> 3.9.5

NETTY_VERSION="4.1.135.Final"
ZOOKEEPER_VERSION="3.9.5"
MAVEN_BASE="https://repo1.maven.org/maven2"

GATK_JAR=$(ls /opt/gatk/gatk-package-*-local.jar 2>/dev/null | head -1)
if [ -z "$GATK_JAR" ]; then
    echo "ERROR: GATK fat JAR not found in /opt/gatk/" >&2
    exit 1
fi

echo "Patching $GATK_JAR ..."
echo "  Netty: -> $NETTY_VERSION"
echo "  ZooKeeper: -> $ZOOKEEPER_VERSION"

NETTY_MODULES=(
    netty-buffer
    netty-codec
    netty-codec-http
    netty-codec-http2
    netty-common
    netty-handler
    netty-resolver
    netty-transport
    netty-transport-native-unix-common
)

WORK_DIR=$(mktemp -d)
DOWNLOAD_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR $DOWNLOAD_DIR" EXIT

# Install zip (needed to repack JAR); unzip is already available
apt-get update -qq && apt-get install -y -qq --no-install-recommends zip >/dev/null 2>&1

echo "Downloading patched JARs..."
for module in "${NETTY_MODULES[@]}"; do
    wget -q -P "$DOWNLOAD_DIR" \
        "${MAVEN_BASE}/io/netty/${module}/${NETTY_VERSION}/${module}-${NETTY_VERSION}.jar"
done

wget -q -P "$DOWNLOAD_DIR" \
    "${MAVEN_BASE}/org/apache/zookeeper/zookeeper/${ZOOKEEPER_VERSION}/zookeeper-${ZOOKEEPER_VERSION}.jar"

echo "Extracting GATK fat JAR..."
cd "$WORK_DIR"
unzip -q "$GATK_JAR"

echo "Replacing netty classes..."
rm -rf io/netty/
for module in "${NETTY_MODULES[@]}"; do
    unzip -qo "${DOWNLOAD_DIR}/${module}-${NETTY_VERSION}.jar" "io/*" -d . 2>/dev/null || true
    unzip -qo "${DOWNLOAD_DIR}/${module}-${NETTY_VERSION}.jar" "META-INF/native-image/*" -d . 2>/dev/null || true
done

echo "Replacing zookeeper classes..."
rm -rf org/apache/zookeeper/
unzip -qo "${DOWNLOAD_DIR}/zookeeper-${ZOOKEEPER_VERSION}.jar" "org/apache/zookeeper/*" -d .

echo "Repacking GATK fat JAR..."
rm "$GATK_JAR"
zip -qr0 "$GATK_JAR" .

# Clean up zip package
apt-get purge -y -qq zip >/dev/null 2>&1
apt-get autoremove -y -qq >/dev/null 2>&1
apt-get clean && rm -rf /var/lib/apt/lists/*

echo "Done. Patched: netty=$NETTY_VERSION, zookeeper=$ZOOKEEPER_VERSION"
