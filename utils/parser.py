import argparse as ag
import os
import json
import yaml
from tabulate import tabulate
import configparser


def get_args(config_file='base.yaml'):
    parser = ag.ArgumentParser(description='Training change detection network')
    configs = yaml.load(open(config_file), Loader=yaml.FullLoader)
    parser.set_defaults(**configs)
     
    args = parser.parse_args()

    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))
    
    return args


def get_parser_with_args_from_json(config_file='configs.json'):
    parser = ag.ArgumentParser(description='Training change detection network')
    config_name = os.path.basename(config_file).split('.')[0]
    
    with open(config_file, 'r') as fin:
        configs = json.load(fin)
        parser.set_defaults(**configs)
        parser.add_argument('--congfig_name', default=config_name, type=str, help='congfigs_name')
        return parser.parse_args()  
    return None