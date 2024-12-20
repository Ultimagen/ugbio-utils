#!/usr/bin/python
# noqa
"""
NOTE: This script is a copy of the original calculate_cost.py script from terra_pipeline.
"""

import argparse
import gzip
import json
import re
import sys
from datetime import timedelta
from io import BytesIO
from math import ceil
from urllib.request import urlopen

# pip install python-dateutil
import dateutil.parser

GCE_MACHINE_TYPES_URL = "http://cloudpricingcalculator.appspot.com/static/data/pricelist.json"
TOTAL_WORKFLOW_COST = 0

CUSTOM_MACHINE_CPU = "CP-COMPUTEENGINE-CUSTOM-VM-CORE"
CUSTOM_MACHINE_RAM = "CP-COMPUTEENGINE-CUSTOM-VM-RAM"
CUSTOM_MACHINE_EXTENDED_RAM = "CP-COMPUTEENGINE-CUSTOM-VM-EXTENDED-RAM"
CUSTOM_MACHINE_CPU_PREEMPTIBLE = "CP-COMPUTEENGINE-CUSTOM-VM-CORE-PREEMPTIBLE"
CUSTOM_MACHINE_RAM_PREEMPTIBLE = "CP-COMPUTEENGINE-CUSTOM-VM-RAM-PREEMPTIBLE"
CUSTOM_MACHINE_EXTENDED_RAM_PREEMPTIBLE = "CP-COMPUTEENGINE-CUSTOM-VM-EXTENDED-RAM-PREEMPTIBLE"
CUSTOM_MACHINE_TYPES = [
    CUSTOM_MACHINE_CPU,
    CUSTOM_MACHINE_RAM,
    CUSTOM_MACHINE_EXTENDED_RAM,
    CUSTOM_MACHINE_CPU_PREEMPTIBLE,
    CUSTOM_MACHINE_RAM_PREEMPTIBLE,
    CUSTOM_MACHINE_EXTENDED_RAM_PREEMPTIBLE,
]
GPU_NVIDIA_TESLA = "GPU_NVIDIA_TESLA"


# load the US pricing for both persistent disk and compute engine
def get_gce_pricing():  # noqa: C901
    response = urlopen(GCE_MACHINE_TYPES_URL)  # noqa: S310
    data = response.read()

    if response.info().get("Content-Encoding") == "gzip":
        buf = BytesIO(data)
        f = gzip.GzipFile(fileobj=buf)
        data = f.read()

    pricing = json.loads(data)

    data = {}
    for k, v in pricing.items():
        if k == "gcp_price_list":
            for k2, v2 in v.items():
                if k2.startswith("CP-COMPUTEENGINE-VMIMAGE"):
                    data[k2.replace("CP-COMPUTEENGINE-VMIMAGE-", "").lower()] = v2["us"]
                if k2 == "CP-COMPUTEENGINE-STORAGE-PD-SSD":
                    data[k2] = v2["us-central1"]
                if k2.startswith("CP-COMPUTEENGINE-LOCAL-SSD"):
                    data[k2] = v2["us-central1"]
                if k2.startswith("CP-COMPUTEENGINE-LOCAL-SSD-PREEMPTIBLE"):
                    data[k2] = v2["us-central1"]
                if k2 == "CP-COMPUTEENGINE-STORAGE-PD-CAPACITY":
                    data[k2] = v2["us-central1"]
                if k2 in CUSTOM_MACHINE_TYPES:
                    data[k2] = v2["us-central1"]
                if k2.startswith("GPU_NVIDIA_TESLA"):
                    data[k2] = v2["us-central1"]

    return data


def extract_machine_type(t):
    if "jes" in t and "machineType" in t["jes"]:
        full_machine = t["jes"]["machineType"]
        if full_machine.startswith("custom"):
            return "custom", full_machine
        else:
            return full_machine.split("/")[1], full_machine
    else:
        return "unknown", "unknown"


def extract_gpu_type(t):
    if "runtimeAttributes" in t and "gpuType" in t["runtimeAttributes"]:
        full_machine = t["runtimeAttributes"]["gpuType"]
        if full_machine.startswith("nvidia"):
            return full_machine
    else:
        return "unknown"


def get_disk_info(metadata):
    if "runtimeAttributes" in metadata and "disks" in metadata["runtimeAttributes"]:
        bootDiskSizeGb = 0.0  # noqa: N806
        if "bootDiskSizeGb" in metadata["runtimeAttributes"]:
            bootDiskSizeGb = float(metadata["runtimeAttributes"]["bootDiskSizeGb"])  # noqa: N806
        # Note - am lumping boot disk in with requested disk.  Assuming boot disk is same type as requested.
        # i.e. is it possible that boot disk is HDD when requested is SDD.
        (name, disk_size, disk_type) = metadata["runtimeAttributes"]["disks"].split()
        return {"size": float(disk_size) + bootDiskSizeGb, "type": "PERSISTENT_" + disk_type}
    else:
        # we can't tell disk size in this case so just return nothing
        return {"size": float(0), "type": "PERSISTENT_SSD"}


def was_preemptible_vm(metadata):
    if "runtimeAttributes" in metadata and "preemptible" in metadata["runtimeAttributes"]:
        pe_count = int(metadata["runtimeAttributes"]["preemptible"])
        attempt = int(metadata["attempt"])

        return attempt <= pe_count
    else:
        # we can't tell (older metadata) so conservatively return false
        return False


def used_cached_results(metadata):
    return "callCaching" in metadata and metadata["callCaching"]["hit"]


def calculate_runtime(call_info, ignore_preempted):
    # get start (start time of VM start) & end time (end time of 'ok') according to metadata

    # give a runtime of 0 for preempted jobs so they have no cost associated with them
    if was_preempted(call_info) and ignore_preempted:
        return 0

    # Accumulate the time spent in the following, which are the ones that we pay for
    pattern = re.compile("Pulling")
    total_amount = timedelta()
    if "executionEvents" in call_info:
        jobs_to_sum = [
            "UserAction",
            "Delocalization",
            "ContainerSetup",
            "Localization",
            "Complete in GCE / Cromwell Poll Interval",
            "^Pulling*",
        ]
        for x in call_info["executionEvents"]:
            if "description" in x:
                y = x["description"]
                if y in jobs_to_sum or pattern.match(y) is not None:
                    start = dateutil.parser.parse(x["startTime"])
                    end = dateutil.parser.parse(x["endTime"])
                    total_amount += end - start
        elapsed = total_amount
    # if we are preempted or if cromwell used previously cached results, we don't even get a start time from JES.
    # if cromwell was restarted, the start time from JES might not have been written to the metadata.
    # in either case, use the Cromwell start time which is earlier but not wrong.
    else:
        start = dateutil.parser.parse(call_info["start"])
        end = dateutil.parser.parse(call_info["end"])
        elapsed = end - start

    # The minimum runtime is 1 minute, after that it's by the second.
    # so if the task ran for 30 seconds, you pay for 1 minute.  If it ran for 1:01 minute, you pay for 1:01 minute
    seconds = elapsed.days * 24 * 60 * 60 + elapsed.seconds
    run_seconds = max(60.0, seconds)
    return run_seconds


def was_preempted(call_info):
    # We treat Preempted and RetryableFailure the same.  The latter is a general case of the former
    return call_info["executionStatus"] in ["Preempted", "RetryableFailure"]


def calculate_cost(metadata, ignore_preempted, only_total_cost, *, print_header, output_file=None):  # noqa: C901, PLR0912, PLR0915
    pipe_cost = 0
    # set up pricing information
    pricing = get_gce_pricing()
    ssd_cost_per_gb_per_month = float(pricing["CP-COMPUTEENGINE-STORAGE-PD-SSD"])
    ssd_cost_per_gb_hour = ssd_cost_per_gb_per_month / (24 * 365 / 12)

    local_ssd_cost_per_gb_per_month = float(pricing["CP-COMPUTEENGINE-LOCAL-SSD"])  # noqa: F841
    local_ssd_cost_per_gb_hour = ssd_cost_per_gb_per_month / (24 * 365 / 12)

    pe_local_ssd_cost_per_gb_per_month = float(pricing["CP-COMPUTEENGINE-LOCAL-SSD-PREEMPTIBLE"])  # noqa: F841
    pe_local_ssd_cost_per_gb_hour = ssd_cost_per_gb_per_month / (24 * 365 / 12)

    hdd_cost_per_gb_per_month = float(pricing["CP-COMPUTEENGINE-STORAGE-PD-CAPACITY"])
    hdd_cost_per_gb_hour = hdd_cost_per_gb_per_month / (24 * 365 / 12)

    disk_costs = {
        "PERSISTENT_SSD": ssd_cost_per_gb_hour,
        "PERSISTENT_HDD": hdd_cost_per_gb_hour,
        "PERSISTENT_LOCAL": local_ssd_cost_per_gb_hour,
        "PE_PERSISTENT_LOCAL": pe_local_ssd_cost_per_gb_hour,
    }

    if print_header and not only_total_cost:
        # print out a header
        header = ",".join(
            [
                "task_name",
                "status",
                "machine_type",
                "cpus",
                "machine_cpus",
                "mem_gbs",
                "total_hours",
                "cpu_cost_per_hour",
                "cpu_cost",
                "mem_cost_per_hour",
                "mem_cost",
                "pe_total_hours",
                "pe_cpu_cost_per_hour",
                "pe_cpu_cost",
                "pe_mem_cost_per_hour",
                "pe_mem_cost",
                "failed_pe_total_hours",
                "failed_pe_cpu_cost",
                "failed_pe_mem_cost",
                "disk_type",
                "disk_size",
                "disk_gb_hours",
                "disk_cost",
                "failed_pe_ssd_gb_hours",
                "failed_pe_ssd_cost",
                "gpu_cost",
                "pe_gpu_cost",
                "total_cost",
            ]
        )

        if output_file:
            with open(output_file, "a") as f:
                print(header, file=f)
        else:
            print(header)

    # iterate through the metadata file for each call
    for k, v in metadata["calls"].items():
        task_name = k

        total_hours = 0
        pe_total_hours = 0
        failed_pe_total_hours = 0
        cpus = 0
        mem_gbs = 0
        gpus = 0
        machine_type = "unknown"
        machine_name = "unknown"
        complete = True
        disk_info = get_disk_info({})
        gpu_type = "unknown"
        for call_info in v:
            # this is a subworkflow, recursively calculate cost on workflow metadata
            if "subWorkflowMetadata" in call_info:
                pipe_cost += calculate_cost(
                    call_info["subWorkflowMetadata"],
                    ignore_preempted,
                    only_total_cost,
                    print_header=False,
                    output_file=output_file,
                )
            # only process things that are not in flight
            elif call_info["executionStatus"] in ["Running", "NotStarted", "Starting"]:
                complete = False
            else:
                if call_info["executionStatus"] in ["Failed"]:
                    complete = False

                if machine_type == "unknown":
                    machine_type, machine_name = extract_machine_type(call_info)

                pe_vm = was_preemptible_vm(call_info)
                disk_info = get_disk_info(call_info)

                run_hours = calculate_runtime(call_info, ignore_preempted) / (60.0 * 60.0)

                # for preemptible VMs, separately tally successful tasks vs ones that were preempted
                if pe_vm:
                    if was_preempted(call_info):
                        # If Compute Engine terminates a preemptible instance less than 10 minutes after it is created,
                        # you are not billed for the use of that virtual machine instance
                        if run_hours < (10.0 / 60.0):
                            run_hours = 0
                        failed_pe_total_hours += run_hours
                    else:
                        pe_total_hours += run_hours
                else:
                    total_hours += run_hours
        # Runtime parameters are the same across all calls; just pull the info from the first one
        if "runtimeAttributes" in v[0]:
            if "cpu" in v[0]["runtimeAttributes"]:
                cpus += int(v[0]["runtimeAttributes"]["cpu"])
            if "memory" in v[0]["runtimeAttributes"]:
                mem_str = v[0]["runtimeAttributes"]["memory"]
                mem_gbs += float(mem_str[: mem_str.index(" ")])
            if "gpuCount" in v[0]["runtimeAttributes"]:
                gpus += int(v[0]["runtimeAttributes"]["gpuCount"])
            if "gpuType" in v[0]["runtimeAttributes"]:
                gpu_type = extract_gpu_type(v[0])

        if complete:
            status = "complete"
        else:
            status = "incomplete"

        machine_cpus = 0
        if machine_type == "custom":
            machine_cpus = int(machine_name.split("-")[1])

        if machine_type != "custom" and machine_type not in pricing:
            if "n2" in machine_type:
                machine_type = machine_type.replace("n2", "n1")
                if machine_type not in pricing:
                    machine_type = "unknown"
            else:
                machine_type = "unknown"

        if machine_type == "unknown":
            cpu_cost_per_hour = 0
            pe_cpu_cost_per_hour = 0
            mem_cost_per_hour = 0
            pe_mem_cost_per_hour = 0
        elif machine_type == "custom":
            cpu_cost_per_hour = pricing[CUSTOM_MACHINE_CPU] * cpus
            pe_cpu_cost_per_hour = pricing[CUSTOM_MACHINE_CPU_PREEMPTIBLE] * cpus
            mem_cost_per_hour = pricing[CUSTOM_MACHINE_RAM] * mem_gbs
            pe_mem_cost_per_hour = pricing[CUSTOM_MACHINE_RAM_PREEMPTIBLE] * mem_gbs
        else:
            cpu_cost_per_hour = pricing[machine_type]
            pe_cpu_cost_per_hour = pricing[machine_type + "-preemptible"]
            mem_cost_per_hour = 0
            pe_mem_cost_per_hour = 0

        gpu_cost_per_hour = 0
        pe_gpu_cost_per_hour = 0
        if gpu_type.startswith("nvidia-tesla-"):
            nvidia_type = gpu_type.split("-")[2]
            full_gpu_type = GPU_NVIDIA_TESLA + "_" + nvidia_type.upper()
            pe_full_gpu_type = GPU_NVIDIA_TESLA + "_" + nvidia_type.upper() + "-PREEMPTIBLE"
            if full_gpu_type in pricing:
                gpu_cost_per_hour = pricing[full_gpu_type] * gpus
                pe_gpu_cost_per_hour = pricing[pe_full_gpu_type] * gpus

        cpu_cost = total_hours * cpu_cost_per_hour
        failed_pe_cpu_cost = failed_pe_total_hours * pe_cpu_cost_per_hour
        pe_cpu_cost = pe_total_hours * pe_cpu_cost_per_hour

        gpu_cost = total_hours * gpu_cost_per_hour
        failed_pe_gpu_cost = failed_pe_total_hours * pe_gpu_cost_per_hour
        pe_gpu_cost = pe_total_hours * pe_gpu_cost_per_hour

        #
        # NOTE -- local ssds have a different price when used in preemptible VMs.
        # However, to implement this all the disk calculations need to be moved from the task level (where it is now)
        # to the call level since each call could be preemptible or not
        # Then we can decide to use PERSISTENT_LOCAL or PE_PERSISTENT_LOCAL
        #
        disk_cost_per_gb_hour = disk_costs[disk_info["type"]]

        if disk_info["type"].endswith("PERSISTENT_LOCAL"):
            local_disk_fix_size = 375
            disk_size = ceil(disk_info["size"] / local_disk_fix_size) * local_disk_fix_size
        else:
            disk_size = disk_info["size"]

        disk_gb_hours = disk_size * (total_hours + pe_total_hours)
        disk_cost = disk_gb_hours * disk_cost_per_gb_hour

        failed_pe_disk_gb_hours = disk_size * failed_pe_total_hours
        failed_pe_disk_cost = failed_pe_disk_gb_hours * disk_cost_per_gb_hour

        mem_cost = total_hours * mem_cost_per_hour
        pe_mem_cost = pe_total_hours * pe_mem_cost_per_hour
        failed_pe_mem_cost = failed_pe_total_hours * pe_mem_cost_per_hour

        total_cost = (
            cpu_cost
            + pe_cpu_cost
            + failed_pe_cpu_cost
            + gpu_cost
            + pe_gpu_cost
            + failed_pe_gpu_cost
            + disk_cost
            + failed_pe_disk_cost
            + mem_cost
            + pe_mem_cost
            + failed_pe_mem_cost
        )

        # accumalate total workflow cost
        global TOTAL_WORKFLOW_COST  # noqa: PLW0603
        TOTAL_WORKFLOW_COST += total_cost
        pipe_cost += total_cost

        if not only_total_cost:
            out = (
                task_name,
                status,
                machine_type,
                cpus,
                machine_cpus,
                mem_gbs,
                total_hours,
                cpu_cost_per_hour,
                cpu_cost,
                mem_cost_per_hour,
                mem_cost,
                pe_total_hours,
                pe_cpu_cost_per_hour,
                pe_cpu_cost,
                pe_mem_cost_per_hour,
                pe_mem_cost,
                failed_pe_total_hours,
                failed_pe_cpu_cost,
                failed_pe_mem_cost,
                disk_info["type"],
                disk_size,
                disk_gb_hours,
                disk_cost,
                failed_pe_disk_gb_hours,
                failed_pe_disk_cost,
                gpu_cost,
                pe_gpu_cost,
                total_cost,
            )
            if output_file:
                with open(output_file, "a") as f:
                    print(",".join(map(str, out)), file=f)
            else:
                print(",".join(map(str, out)))

    return pipe_cost


def compare(old, new):
    """Fail when NEW total exceeds OLD total by > 5%."""

    def total(cost_file):
        with open(cost_file) as input:  # noqa: A001
            lines = input.readlines()
        for line in lines:
            fields = line.split()
            if len(fields) == 3 and fields[0] == "Total" and fields[1] == "Cost:":  # noqa: PLR2004
                return int(float(fields[2]) * 10000) / 10000.0
        return None

    old_cost = total(old)
    new_cost = total(new)
    if old_cost and new_cost:
        more = new_cost - old_cost
        percent = more * 100 / old_cost
        if more > 0.0:
            print(f"Cost has increased by ${more} ({percent}%): from ${old_cost} to ${new_cost}")
            sys.exit(0)
        else:
            if more < 0.0:
                down = percent * -1
                print(f"Cost has decreased by ${more} ({down}%): from ${old_cost} to ${new_cost}")
            else:
                print(f"Cost is the same: ${new_cost}")
            print("Everything is awesome!")
            sys.exit(0)
    sys.exit(f"One or both of the calculated costs is 0!  WTF?: old ({old_cost}) new ({new_cost})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ignore_preempted", dest="ignore_preempted", action="store_true", help="ignore preempted tasks"
    )
    parser.add_argument(
        "--only_total",
        dest="only_total_cost",
        action="store_true",
        help="print total cost of the workflow instead of the tsv per task costs",
    )
    either = parser.add_mutually_exclusive_group(required=True)
    either.add_argument("-m", "--metadata", dest="metadata", help="metadata file to calculate cost on")
    either.add_argument("--compare", nargs=2, help="compare old to new cost output")

    args = parser.parse_args()

    if args.metadata:
        with open(args.metadata) as data_file:
            metadata = json.load(data_file)
        calculate_cost(metadata, args.ignore_preempted, args.only_total_cost, print_header=True)
        if args.only_total_cost:
            print("Total Cost: " + str(TOTAL_WORKFLOW_COST))
    else:
        old, new = args.compare
        compare(old, new)


if __name__ == "__main__":
    main()
