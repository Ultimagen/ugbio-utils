#!/bin/env python3

"""
computes the cost of a workflow run and individual tasks
source: https://github.com/aws-samples/amazon-omics-tutorials/blob/32757755f03e40f788b5dce95eaf09286bc65ed5/utils/scripts/compute_pricing.py
"""

from argparse import ArgumentParser
import json

import boto3
import requests
from datetime import datetime, timedelta, UTC

parser = ArgumentParser()
parser.add_argument('--region', type=str, help="AWS region to use")
parser.add_argument('--offering', type=str, help="path to pricing offer JSON")
parser.add_argument('run_id', type=str, help="HealthOmics workflow run-id to analyze")


def get_pricing(offering=None, client=None):
    if offering:
        # user specified offering
        with open(offering, 'r') as file:
            offering = json.load(file)
    else:
        # retrieve offring using Bulk API
        if not client:
            client = boto3.client('omics')

        region = client.meta.region_name
        response = requests.get(
            f'https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonOmics/current/{region}/index.json')
        if not response.ok:
            response.raise_for_status()

        offering = response.json()

    compute_offering = {
        product[0]: product[1]
        for product in offering['products'].items() if product[1]['productFamily'] == 'Compute'
    }

    for key in compute_offering:
        offer_term = list(offering['terms']['OnDemand'][key].values())[0]
        price_dimensions = list(offer_term['priceDimensions'].values())[0]
        compute_offering[key]['priceDimensions'] = price_dimensions

    # set resourceType as the primary hash key
    pricing = {
        instance['attributes']['resourceType']: instance
        for instance in compute_offering.values()
        if instance['attributes'].get('resourceType')
    }

    return pricing


def get_run_info(run_id, client=None):
    if not client:
        client = boto3.client('omics')

    run = client.get_run(id=run_id)
    run.update({"duration": (run['stopTime'] if 'stopTime' in run else datetime.now(UTC)) - run['startTime']})

    response = client.list_run_tasks(id=run_id)
    tasks = response['items']
    while response.get('nextToken'):
        response = client.list_run_tasks(id=run_id, startingToken=response.get('nextToken'))
        tasks += response['items']

    def calc_task_duration(task):
        stop_time = task['stopTime'] if 'stopTime' in task else datetime.now(UTC)
        start_time = task[
            'startTime'] if 'startTime' in task else stop_time  # when task is CANCELLED sometime there's no startTime
        return stop_time - start_time

    tasks = [{**task, "duration": calc_task_duration(task)} for task in tasks]

    del run['ResponseMetadata']

    run['tasks'] = tasks
    return run


MINIMUM_STORAGE_CAPACITY_GIB = 1200


def get_run_cost(run_id, storage_gib=MINIMUM_STORAGE_CAPACITY_GIB, client=None, offering=None):
    if not client:
        client = boto3.client('omics')

    pricing = get_pricing(offering=offering, client=client)
    STORAGE_USD_PER_GIB_PER_HR = float(pricing['Run Storage']['priceDimensions']['pricePerUnit']['USD'])

    run = get_run_info(run_id, client=client)
    run_duration_hr = run['duration'].total_seconds() / 3600

    task_costs = []
    for task in run['tasks']:
        if not task.get('gpus'):
            task['gpus'] = 0
        usd_per_hour = float(pricing[task['instanceType']]['priceDimensions']['pricePerUnit']['USD'])
        duration_hr = task['duration'].total_seconds() / 3600
        task_costs += [{
            "name": task['name'],
            "resources": {
                "cpus": task['cpus'],
                "memory_gib": task['memory'],
                "gpus": task['gpus']
            },
            "duration_hr": duration_hr,
            "instance": task['instanceType'],
            "usd_per_hour": usd_per_hour,
            "cost": duration_hr * usd_per_hour
        }]

    if not run.get('storageCapacity'):
        # assume the default storage capacity of 1200 GiB
        pass
    else:
        storage_gib = run['storageCapacity']

    if storage_gib < MINIMUM_STORAGE_CAPACITY_GIB:
        storage_gib = MINIMUM_STORAGE_CAPACITY_GIB

    storage_cost = run_duration_hr * storage_gib * STORAGE_USD_PER_GIB_PER_HR
    total_task_costs = sum([tc['cost'] for tc in task_costs])

    return {
        "info": {
            "runId": run['id'],
            "name": run['name'],
            "workflowId": run['workflowId']
        },
        "total": storage_cost + total_task_costs,
        "cost_detail": {
            "storage_cost": {
                "run_duration_hr": run_duration_hr,
                "storage_gib": storage_gib,
                "usd_per_hour": STORAGE_USD_PER_GIB_PER_HR,
                "cost": storage_cost
            },
            "total_task_cost": total_task_costs,
            "task_costs": task_costs
        }
    }


if __name__ == "__main__":
    args = parser.parse_args()
    session = boto3.Session(region_name=args.region)

    cost = get_run_cost(args.run_id, client=session.client('omics'), offering=args.offering)
    print(json.dumps(cost, indent=4, default=str))
