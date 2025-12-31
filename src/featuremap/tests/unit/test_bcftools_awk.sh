#!/bin/bash
# Test script to diagnose bcftools→AWK pipeline issues

set -euo pipefail

# Configuration (adjust these)
VCF="/data/Runs/SRSNV/Pa_46/Pa_46_333_LuNgs_08.random_sample.featuremap.vcf.gz"
REGION="chr1:1-10000000"  # Test first 10Mbp
AWK_SCRIPT="/home/ubuntu/ugbio-utils/src/featuremap/ugbio_featuremap/explode_lists.awk"

echo "Testing bcftools→AWK pipeline for region: $REGION"
echo "================================================"

# Count variants in region
echo "1. Counting variants in region..."
VARIANT_COUNT=$(bcftools view -H -r "$REGION" "$VCF" | wc -l)
echo "   Found $VARIANT_COUNT variants"

# Test bcftools query output
echo ""
echo "2. Testing bcftools query (first 5 lines)..."
bcftools query -r "$REGION" \
  -f '%CHROM\t%POS\t%ID\t%QUAL\t%REF\t%ALT\t%INFO/gnomAD_AF\t%INFO/UG_HCR\t[%DP\t%BCSQ\t%RL\t%INDEX]\n' \
  "$VCF" | head -5

# Count bcftools output lines
echo ""
echo "3. Counting bcftools query output lines..."
BCFTOOLS_LINES=$(bcftools query -r "$REGION" \
  -f '%CHROM\t%POS\t%ID\t%QUAL\t%REF\t%ALT\t%INFO/gnomAD_AF\t%INFO/UG_HCR\t[%DP\t%BCSQ\t%RL\t%INDEX]\n' \
  "$VCF" | wc -l)
echo "   bcftools output: $BCFTOOLS_LINES lines"

# Test AWK explosion
echo ""
echo "4. Testing bcftools→AWK pipeline..."
# List columns are: DP(8-scalar), BCSQ(9-list), RL(10-list), INDEX(11-list)
# List indices (0-based): 9, 10, 11
AWK_LINES=$(bcftools query -r "$REGION" \
  -f '%CHROM\t%POS\t%ID\t%QUAL\t%REF\t%ALT\t%INFO/gnomAD_AF\t%INFO/UG_HCR\t[%DP\t%BCSQ\t%RL\t%INDEX]\n' \
  "$VCF" | \
  awk -v list_indices="9,10,11" -f "$AWK_SCRIPT" | \
  wc -l)
echo "   AWK output: $AWK_LINES lines"

# Calculate explosion factor
if [ "$BCFTOOLS_LINES" -gt 0 ]; then
  EXPLOSION_FACTOR=$(echo "scale=2; $AWK_LINES / $BCFTOOLS_LINES" | bc)
  echo "   Explosion factor: ${EXPLOSION_FACTOR}x (AWK lines / bcftools lines)"
fi

# Show first few exploded rows
echo ""
echo "5. First 10 exploded rows:"
bcftools query -r "$REGION" \
  -f '%CHROM\t%POS\t%ID\t%QUAL\t%REF\t%ALT\t%INFO/gnomAD_AF\t%INFO/UG_HCR\t[%DP\t%BCSQ\t%RL\t%INDEX]\n' \
  "$VCF" | \
  awk -v list_indices="9,10,11" -f "$AWK_SCRIPT" | \
  head -10

# Check for errors in pipeline
echo ""
echo "6. Testing for pipeline errors..."
set +e  # Don't exit on error
bcftools query -r "$REGION" \
  -f '%CHROM\t%POS\t%ID\t%QUAL\t%REF\t%ALT\t%INFO/gnomAD_AF\t%INFO/UG_HCR\t[%DP\t%BCSQ\t%RL\t%INDEX]\n' \
  "$VCF" 2>bcftools_err.log | \
  awk -v list_indices="9,10,11" -f "$AWK_SCRIPT" 2>awk_err.log > /dev/null

BCFTOOLS_RC=$?
AWK_RC=${PIPESTATUS[1]}

if [ -s bcftools_err.log ]; then
  echo "   ⚠️  bcftools stderr:"
  cat bcftools_err.log | head -20
else
  echo "   ✓ bcftools: no errors"
fi

if [ -s awk_err.log ]; then
  echo "   ⚠️  AWK stderr:"
  cat awk_err.log | head -20
else
  echo "   ✓ AWK: no errors"
fi

rm -f bcftools_err.log awk_err.log

echo ""
echo "Summary:"
echo "--------"
echo "Variants: $VARIANT_COUNT"
echo "bcftools output: $BCFTOOLS_LINES lines"
echo "AWK exploded output: $AWK_LINES lines"
if [ "$BCFTOOLS_LINES" -gt 0 ]; then
  echo "Expected reads per variant: ~$(echo "scale=1; $AWK_LINES / $BCFTOOLS_LINES" | bc)"
fi
