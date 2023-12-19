from argparse import ArgumentParser
import pandas as pd
import os
from os.path import join as pjoin
import json

def main(raw_args=None):
    parser = ArgumentParser()
    parser.add_argument("cost_json", help="cost.json file generated from compute_pricing script")

    args = parser.parse_args(raw_args)

    cost_json = json.load(open(args.cost_json))
    tasks_costs = cost_json['cost_detail']['task_costs']

    cost_list = []
    duration_hr_list = []
    instance_list = []
    name_list = []
    resources_list = []
    usd_per_hour_list = []

    for i in range(0, len(tasks_costs)):
        cost_list.append(cost_json['cost_detail']['task_costs'][i]['cost'])
        duration_hr_list.append(cost_json['cost_detail']['task_costs'][i]['duration_hr'])
        instance_list.append(cost_json['cost_detail']['task_costs'][i]['instance'])
        name_list.append(cost_json['cost_detail']['task_costs'][i]['name'])
        resources_list.append(cost_json['cost_detail']['task_costs'][i]['resources'])
        usd_per_hour_list.append(cost_json['cost_detail']['task_costs'][i]['usd_per_hour'])

    dict = {'cost': cost_list,
            'duration_hr': duration_hr_list,
            'instance': instance_list,
            'name': name_list,
            'resources': resources_list,
            'usd_per_hour': usd_per_hour_list
            }
    df_tasks_costs = pd.DataFrame(dict)

    df_tasks_costs.to_csv(os.path.splitext(args.cost_json)[0]+'.csv',index=False)

if __name__ == '__main__':
    main()
