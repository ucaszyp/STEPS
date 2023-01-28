from .robot_car import RobotCarSequence
from .multiple_robotcar import MultipleRobotCar
from .common import *
from .nuscenes import nuScenesSequence
from .multiple_nuscenes import MultipleNuScenes

def build_dataset(data_option):
    """
    Return corresponding dataset to given dataset option
    :param data_option:
    :return:
    """
    dataset_type = data_option['type']
    if dataset_type == 'robotcar':
        rc_opt = data_option['robotcar']
        dataset = RobotCarSequence(rc_opt['root_dir'], data_option['frame_ids'], True, rc_opt['down_scale'],
                                   rc_opt['num_out_scales'], rc_opt['gen_equ'], rc_opt['equ_limit'], rc_opt['resize'])
    elif dataset_type == 'multiple_robotcar':
        mrc_opt = data_option['multiple_robotcar']
        dataset = MultipleRobotCar(mrc_opt['subsets'], data_option.frame_ids, mrc_opt['master'], True,
                                   mrc_opt['down_scale'], mrc_opt['num_out_scales'], mrc_opt['gen_equ'],
                                   mrc_opt['shuffle'], equ_limit=mrc_opt['equ_limit'])
    elif dataset_type == 'nuscenes':
        ns_opt = data_option['nuscenes']
        dataset = nuScenesSequence(ns_opt['weather'], data_option['frame_ids'], True, ns_opt['down_scale'],
                                   ns_opt['num_out_scales'], ns_opt['gen_equ'], ns_opt['equ_limit'], ns_opt['resize'])
    elif dataset_type == 'multiple_nuscenes':
        mns_opt = data_option['multiple_nuscenes']
        dataset = MultipleNuScenes(mns_opt['subsets'], data_option.frame_ids, mns_opt['master'], True,
                                   mns_opt['down_scale'], mns_opt['num_out_scales'], mns_opt['gen_equ'],
                                   mns_opt['shuffle'])
    elif dataset_type == 'carlaepe':
        ce_opt = data_option['carlaepe']
        dataset = CARLAEPESequence(ce_opt['weather'], data_option['frame_ids'], True, ce_opt['down_scale'],
                                   ce_opt['num_out_scales'], ce_opt['gen_equ'], ce_opt['equ_limit'], ce_opt['resize'])
    elif dataset_type == 'multiple_carlaepe':
        mce_opt = data_option['multiple_carlaepe']
        dataset = MultipleCARLAEPE(mce_opt['subsets'], data_option.frame_ids, mce_opt['master'], True,
                                   mce_opt['down_scale'], mce_opt['num_out_scales'], mce_opt['gen_equ'],
                                   mce_opt['shuffle'])
    else:
        raise ValueError('Unknown dataset type: {}.'.format(dataset_type))

    return dataset
