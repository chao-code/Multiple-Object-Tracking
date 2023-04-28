import os
import yaml
import trackeval


def evaluator():
    with open(f'tracker/config_files/MOT17_train.yaml', 'r') as f:  # TODO
        cfgs = yaml.load(f, Loader=yaml.FullLoader)

    seqs = []
    folder_path = os.listdir('./results/GT_results')
    for folder in folder_path:
        for root, dirs, files in os.walk(os.path.join('./results/GT_results', folder)):
            for file in files:
                filename, extension = os.path.splitext(file)
                if filename not in seqs:
                    seqs.append(filename)

    seqs = sorted(seqs)
    seqs = [seq for seq in seqs if seq not in cfgs['IGNORE_SEQS']]

    # 获取trackeval包中评估的默认配置
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    # 获取trackeval包中数据集的默认配置
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    yaml_dataset_config = cfgs['TRACK_EVAL']  # read yaml file to read TrackEval configs
    # make sure that seqs is same as 'SEQ_INFO' in yaml
    # delete key in 'SEQ_INFO' which is not in seqs
    seqs_in_cfgs = list(yaml_dataset_config['SEQ_INFO'].keys())
    for k in seqs_in_cfgs:
        if k not in seqs:
            yaml_dataset_config['SEQ_INFO'].pop(k)
    # assert len(yaml_dataset_config['SEQ_INFO'].keys()) == len(seqs)

    for k in default_dataset_config.keys():
        if k in yaml_dataset_config.keys():  # if the key need to be modified
            default_dataset_config[k] = yaml_dataset_config[k]

    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)

if __name__ == '__main__':
    evaluator()