import numpy as np
from scipy import stats
import time
import random
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from math import log
import statsmodels.api as sm
import ugbio_core.flow_format.flow_based_pileup as fbp
from scipy.stats import combine_pvalues
from ugbio_core.flow_format.flow_based_pileup import get_hmer_qualities_from_pileup_element
import sys
import pysam
import math

def calc_exp(eyze):
    ex = 0
    for i in range(len(eyze)):
        ex+= i*eyze[i]
    return ex


def get_max_nuc(nuc_list):
    nuc_dc = {}
    for nuc in nuc_list:
        if nuc in nuc_dc:
            nuc_dc[nuc] += 1
        else:
            nuc_dc[nuc] = 1
    max_count = 0
    max_nuc = None
    for x in nuc_dc:
        if nuc_dc[x] > max_count:
            max_count = nuc_dc[x]
            max_nuc = x
    return max_nuc

def filter_results(results, nuc):
    filtered_results = []
#    cycle_dc = {}
    for res in results:
        if res[0] != nuc or res[4] or not res[5]:
            continue
#        k = (res[2], res[3])
#        if k in cycle_dc:
#            continue
#        cycle_dc[k] = 0
        expect = calc_exp(res[1])
        res[1].sort()
        max_conf = res[1][-1]
        high_conf = res[1][-3]
        filtered_results.append((expect, max_conf, high_conf, res[2], res[3]))
    return filtered_results

def calc_parameters(reads):
    remainder_list = [abs(x[0]-round(x[0])) for x in reads]
    remainder_list.sort()
    remainder = remainder_list[len(remainder_list)*2//3]
    exp_split = (remainder, 0.5, 1-remainder)
    if len(remainder_list) > 1000:
        high_conf_list = [x[2] for x in reads]
        high_conf_list.sort()
        high_conf = high_conf_list[len(high_conf_list)*2//3]
        cycle = 200
        if len(remainder_list)>3000:
            remainder2 = remainder_list[len(remainder_list)*3//7]
            exp_split = (remainder2, remainder, 0.5, 1-remainder, 1 - remainder2)
    else:
        cycle = 5000
        high_conf = 1.1
#    cycles = [[], []]
#    high_conf = [[], []]
#    for i, read in enumerate(reads):
#        cycles[remainder_list[i] < remainder].append(read[4])
#        high_conf[remainder_list[i] < remainder].append(read[2])
#    try:
#        cycles[0].sort()
#        cycles[1].sort()
#        cycle1 = cycles[0][len(cycles[0]) // 2]
#        cycle2 = cycles[1][len(cycles[1]) // 2]
#        high_conf[0].sort()
#        high_conf[1].sort()
#        high_conf1 = high_conf[0][len(high_conf[0]) // 2]
#        high_conf2 = high_conf[1][len(high_conf[1]) // 2]
#        return [(remainder, 0.5, 1-remainder) ,[(cycle1, high_conf1), (cycle2, high_conf2), (cycle2, high_conf2), (cycle1, high_conf1)]]
#    except:
    return [exp_split , (high_conf, cycle)]

def get_cell(read, params, cell_shift):
    split_values = params[0]
    expect = read[0]
    floor_expect = int(expect)
    cell = floor_expect * cell_shift
    remainder = expect - floor_expect
    for i in split_values:
        if remainder < i:
            break
        cell += 4
    cell += read[3]
    cell += 2*(read[2] < params[1][0] and read[4] < params[1][1])
#    cell += 2*(read[2]>params[1][ind][1])
#    cell += 2*(read[4]>params[1][ind][0])
    return cell

def get_pval(exps, groups, w):
    y = np.asarray(exps, dtype=float)
    g = np.asarray(groups, dtype=float)  # 0 or 1
    w = np.asarray(w, dtype=float)

    X = sm.add_constant(g)  # intercept + group effect
    res = sm.WLS(y, X, weights = w).fit()
    p_val  = float(res.pvalues[1])
    length = len(exps)
    exp_position = [(exps[x], x) for x in range(length)]
    exp_position.sort()
    added_weights_max = np.asarray(w, dtype=float)
    added_weights_min = np.asarray(w, dtype=float)
    for i in range(length):
        added_weights_max[exp_position[i][1]] *= (((i+1.0)/length)**2
                                             ) + 1.0
        added_weights_min[exp_position[i][1]] *= (((length-i)*1.0/length)**2) + 1.0
    res = sm.WLS(y, X, weights = added_weights_max).fit()
    p_val = min(p_val, float(res.pvalues[1]))
    res = sm.WLS(y, X, weights = added_weights_min).fit()
    p_val = min(p_val, float(res.pvalues[1]))
    return p_val

# 0.35, [0.007, 0.5, 1-0.007], 0.0064, 200)
def variant_calling(vcf_file, normal_file, tumor_file, vcf_out_file, min_hmer=6, zamir = 0.5, exp_shift_tries = [-4, -3, -2, -1, 1, 2, 3, 4]):
    normal=pysam.AlignmentFile(normal_file)
    tumor=pysam.AlignmentFile(tumor_file)
    f = pysam.VariantFile(vcf_file)
    f.header.info.add("normal_exp", number=1, type="Float", description="Normal hmer size expectancy")
    f.header.info.add("tumor_exp", number=1, type="Float", description="Tumor hmer sizeexpectancy")
    f.header.info.add("normal_cvg", number=1, type="Integer", description="Number of normal reads")
    f.header.info.add("tumor_cvg", number=1, type="Integer", description="Number of tumor reads")
    f.header.info.add("ins_size", number=1, type="Integer", description="Best insertion size")
    f.header.info.add("ttest_score", number=1, type="Float", description="Score from ttest")
    f.header.info.add("likely_score", number=1, type="Float", description="Score from likelyhood")
    f.header.info.add("mixture", number=1, type="Float", description="Best mixture size")
    fo = pysam.VariantFile(vcf_out_file, "wz", header=f.header.copy())
    for rec in f.fetch():
        if rec.info["VARIANT_TYPE"] != 'h-indel' or rec.info["X_HIL"][0] < min_hmer:
            fo.write(rec)
            continue
        chrom = rec.contig
        pos = rec.pos + rec.info["X_HIL"][0]//2 #Ask Ilya 
        try:
            exps = []
            groups = []
            weights = []
            p = normal.pileup(chrom, pos - 1, pos, truncate=True, min_base_quality=0, min_mapping_quality=0)
            pileup = next(p)
            reads = pileup.pileups
            results = [ get_hmer_qualities_from_pileup_element(x) for x in reads]
            nuc = get_max_nuc([x[0] for x in results])
            filtered_results = filter_results(results, nuc)
            normal_cvg = len(filtered_results)
            params = calc_parameters(filtered_results)
            cell_shift = 2*2*len(params[0])
            num_cells = cell_shift * 21
            original_counts = [zamir] * num_cells
            extra_counts = [[zamir] * num_cells for x in exp_shift_tries]
            for read in filtered_results:
                exps.append(read[0])
                groups.append(0)
                certainty = read[1]
                weights.append(certainty)
                cell = get_cell(read, params, cell_shift)
                original_counts[cell] += 1
                for x in range(len(exp_shift_tries)):
                    curr_cell = cell + cell_shift*exp_shift_tries[x]
                    if curr_cell>=0 and curr_cell < num_cells:
                        extra_counts[x][curr_cell] += 1
            sum_counts = sum(original_counts)
            normal_exp = sum(exps)/len(exps)
            original_probs = [x/sum_counts for x in original_counts]
            extra_probs = []
            for count in extra_counts:
                sum_count = sum(count)
                extra_probs.append([x/sum_count for x in count])
            original_likelyhood = 0
            h0_probs = []
            h1_probs = [[] for x in range(len(exp_shift_tries))]
            p = tumor.pileup(chrom, pos - 1, pos, truncate=True, min_base_quality=0, min_mapping_quality=0)
            pileup = next(p)
            reads = pileup.pileups
            results = [ get_hmer_qualities_from_pileup_element(x) for x in reads]
            filtered_results = filter_results(results, nuc)
            tumor_cvg = len(filtered_results)
            if tumor_cvg==0:
                print(f'failed on {chrom} position {pos} - tumor cvg is 0')
                fo.write(rec)
                continue
            tot_number = 0
            tot_exp = 0
            for read in filtered_results:
                tot_number+=1
                tot_exp += read[0]
                exps.append(read[0])
                groups.append(1)
                certainty = read[1]
                weights.append(certainty)
                cell = get_cell(read, params, cell_shift)
                h0_probs.append(original_probs[cell])
                original_likelyhood += log(original_probs[cell])
                for i in range(len(exp_shift_tries)):
                    h1_probs[i].append(extra_probs[i][cell])
            tumor_exp = tot_exp/tot_number
            ttest_pval = max(get_pval(exps, groups, weights), 0.0000000000001)
            h0_probs = np.asarray(h0_probs, dtype=float)
            h1_mixtures = []
            for i in range(len(exp_shift_tries)):
                try:
                    h1_probs[i] = np.asarray(h1_probs[i], dtype=float)
                    pi = 0.05
                    for _ in range(6):
                        denom = pi * h1_probs[i] + (1.0 - pi) * h0_probs
                        mon = pi * h1_probs[i]
                        r = mon / denom
                        pi_new = float(r.mean())
                        pi = min(max(pi_new, 0.001), 0.999)    
                    h1_mixtures.append(pi)
                except:
                    h1_mixtures.append(0.002)
            extra_likelyhoods = []
            for i in range(len(exp_shift_tries)):
                likely = 0
                for j in range(len(h0_probs)):
                    likely += log((1-h1_mixtures[i]) * h0_probs[j] + h1_mixtures[i] * h1_probs[i][j])
                extra_likelyhoods.append(likely)
            extra_score = max(extra_likelyhoods)
            extra_shift = extra_likelyhoods.index(extra_score)
            mixture = h1_mixtures[extra_shift]
            tmp_score = (-0.6*max(original_likelyhood - extra_score + log(len(exp_shift_tries)), -400.0)) / log(10)
            tmp_score = min(20.0, tmp_score)
            pval_score = -log(ttest_pval)/log(10)
            likely_score = -log(ttest_pval)/log(10)
            if tmp_score > 0:
                likely_score += tmp_score
            rec.info["normal_exp"] = round(normal_exp, 4)
            rec.info["tumor_exp"] = round(tumor_exp, 4)
            rec.info["normal_cvg"] = normal_cvg
            rec.info["tumor_cvg"] = tumor_cvg
            rec.info["ins_size"] = exp_shift_tries[extra_shift]
            rec.info["ttest_score"] = round(pval_score, 4)
            rec.info["likely_score"] = round(tmp_score, 4)
            rec.info["mixture"] = round(mixture, 4)
            fo.write(rec)
        except:
            print(f'failed on {chrom} position {pos}')
            fo.write(rec)
    f.close()
    fo.close()
min_hmer = 6
if len(sys.argv) > 5:
    min_hmer = int(sys.argv[5])
variant_calling(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], min_hmer)
