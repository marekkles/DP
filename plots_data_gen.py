# %%
from lightning.evaluation import *
import os
import pickle
import numpy as np
import scipy.stats as stats

scores = {}
qualities = {}
datasets_names = {
    'iris_verification_NDCSI2013_01_05': 'Notre Damme 2013',
    'iris_verification_nd_0405_01_01': 'Notre Damme 0405',
    'iris_verification_inno_keymaker_01_01': 'Innovatrics',
    'iris_verification_iitd_01_01': 'IITD',
}
comments = {
    'imOXXb' : ('CrFiqaNet-iresnet18-CrFiqaLoss-imOXXb'              , 'Trained well, gives results, certainty ratio'),
    'GzGNXb' : ('CrFiqaNet-iresnet50-CrFiqaLoss-GzGNXb'              , 'Trained well, gives results, certainty ratio'),
    'H1SWXb' : ('CrFiqaNet-magiresnet18-CrFiqaLoss-H1SWXb'           , 'Did not train, loss to high number'),
    '0IRWXb' : ('CrFiqaNet-mobilenetv3_large-CrFiqaLoss-0IRWXb'      , 'Trained well, certainty ratio'),
    '0G8VXb' : ('DfsNet-0G8VXb'                                      , 'Refernce with 640x480 input'),
    #'hdpNXb' : ('DfsNet-hdpNXb'                                      , 'Reference with 224x244 input'),
    'XSkYXb' : ('DfsNet-XSkYXb'                                      , 'CX2 Reference with 640x480 input'),
    #'R4kYXb' : ('DfsNet-R4kYXb'                                      , 'CX2 Reference with 224x244 input'),
    '3wDNXb' : ('PfeNet-iresnet50-3wDNXb'                            , 'PFE with ArcFace'),
    'RyENXb' : ('PfeNet-iresnet50-RyENXb'                            , 'PFE with MagFace'),
    '6MrZXb' : ('PfeNet-magiresnet18-6MrZXb'                         , 'PFE with iresnet18 MagFace'),
    '8KuZXb' : ('PfeNet-magiresnet50-8KuZXb'                         , 'PFE with iresnet50 ArcFace'),
    'QTsZXb' : ('PfeNet-magiresnet50-QTsZXb'                         , 'PFE with iresnet50 MagFace'),
    'cWqZXb' : ('PfeNet-mobilenetv3_large-cWqZXb'                    , 'PFE with MobileNetV3-Large MagFace'),
    'YuBXXb' : ('RecognitionNet-iresnet18-MagFaceLoss-YuBXXb'        , 'iResNet18, crop, without fully connected layer'),
    '0g9LXb' : ('RecognitionNet-iresnet50-ArcFaceLoss-0g9LXb'        , 'Unwrap baseline with ArcFace, without fully connected layer'),
    'UVyKXb' : ('RecognitionNet-iresnet50-ArcFaceLoss-UVyKXb'        , 'Crop baseline with ArcFace, without fully connected layer'),
    'DaqKXb' : ('RecognitionNet-iresnet50-MagFaceLoss-DaqKXb'        , 'Crop baseline with MagFace, without fully connected layer'),
    'ciWLXb' : ('RecognitionNet-iresnet50-MagFaceLoss-ciWLXb'        , 'Unwrap baseline with MagFace, without fully connected layer'),
    '7B0TXb' : ('RecognitionNet-magiresnet18-MagFaceLoss-7B0TXb'     , 'Showcase of wrong schedule settings'),
    'xjxTXb' : ('RecognitionNet-magiresnet18-MagFaceLoss-xjxTXb'     , 'Showcase of good schedule setting'),
    'RMdMXb' : ('RecognitionNet-magiresnet50-ArcFaceLoss-RMdMXb'     , 'Unwraped ArcFace'),
    'arzKXb' : ('RecognitionNet-magiresnet50-ArcFaceLoss-arzKXb'     , 'Croped ArcFace'),
    '931RXb' : ('RecognitionNet-magiresnet50-MagFaceLoss-931RXb'     , 'No augment'),
    'JnGKXb' : ('RecognitionNet-magiresnet50-MagFaceLoss-JnGKXb'     , 'Crop baseline'),
    'ihfMXb' : ('RecognitionNet-magiresnet50-MagFaceLoss-ihfMXb'     , 'Unwrap baseline'),
    'vfFRXb' : ('RecognitionNet-magiresnet50-MagFaceLoss-vfFRXb'     , 'Shear + translate'),
    'PUXTXb' : ('RecognitionNet-mobilenetv3_large-MagFaceLoss-PUXTXb', 'Small MobileNetV3, baseline crop, without fully connected layer'),
    'BnkOXb' : ('SddFiqaNet-iresnet50-BnkOXb'                        , 'SDDFIQA with labels from DaqKXb, with labels from iResNet50 crop without fc'),
    'UeIWXb' : ('SddFiqaNet-iresnet50-UeIWXb'                        , 'SDDFIQA with labels from PUXTXb, with labels fomr MobileNetV3'),
    'UnMWXb' : ('SddFiqaNet-iresnet50-UnMWXb'                        , 'SDDFIQA with labels from 7B0TXb, wrong magiresenet18'),
    'e9KWXb' : ('SddFiqaNet-iresnet50-e9KWXb'                        , 'SDDFIQA with labels from xjxTXb, small magiresnet18'),
    'wJJWXb' : ('SddFiqaNet-iresnet50-wJJWXb'                        , 'SDDFIQA with labels from GOTTXb, untrained mobilenetv3'),
}

def get_data(type, run, runs_root, dataset):
    global comments
    path = os.path.join(runs_root, comments[run][0], f'{type}-{comments[run][0]}-{dataset}.pickle')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
def get_det_for_run(run, runs_root, dataset):
    global datasets
    global scores
    if not (dataset+'+'+run) in scores:
        embeddings = get_data('embedding', run, runs_root, dataset)
        sc_original = pairs_impostor_scores(datasets[dataset]['pairs'], datasets[dataset]['impostors'], embeddings, 'cosine')
        labels, sc, pairs = generate_labels_scores(sc_original['pairs'], sc_original['impostors'])
        sc = -sc
        scores[dataset+'+'+run] = sc_original
    else:
        sc_original = scores[dataset+'+'+run]
        labels, sc, pairs = generate_labels_scores(sc_original['pairs'], sc_original['impostors'])
    return det_curve(labels, sc)
def get_magnitude(run, runs_root, dataset):
    global datasets
    embeddings = get_data('embedding', run, runs_root, dataset)
    magnitudes = {}
    for id in embeddings:
        magnitudes[id] = np.linalg.norm(embeddings[id])
    return magnitudes
def pfe_quality(log_sigma):
    return stats.hmean(log_sigma)

# %%
def main(runs_root = 'runs_mnt', comments_path = 'comments.pickle', datasets_path = 'datasets.pickle', scores_path = 'scores.pickle', quality_path = 'qualities.pickle'):
    global datasets
    global comments
    global scores

    with open(comments_path, 'wb') as f:
        pickle.dump(comments, f)

    with open(datasets_path, 'rb') as f:
        datasets = pickle.load(f)

    baselines = [
        'YuBXXb',
        '0g9LXb',
        'UVyKXb',
        'DaqKXb',
        'ciWLXb',
        '7B0TXb',
        'xjxTXb',
        'RMdMXb',
        'arzKXb',
        '931RXb',
        'JnGKXb',
        'ihfMXb',
        'vfFRXb',
        'PUXTXb',
        'imOXXb',
        'GzGNXb',
        '0IRWXb',
    ]

    dets = {}
    magface_dets = {}

    for d in datasets:
        for run in baselines:
            print(f'Loading: {d} {run}')
            dets[d+'+'+run] = get_det_for_run(run, runs_root, d)
    with open(scores_path, 'wb') as f:
        pickle.dump(scores, f)

    
    baselines = [
    'imOXXb',
    'GzGNXb',
    'H1SWXb',
    '0IRWXb',
    '0G8VXb',
    'hdpNXb',
    '3wDNXb',
    'RyENXb',
    'YuBXXb',
    '0g9LXb',
    'UVyKXb',
    'DaqKXb',
    'ciWLXb',
    '7B0TXb',
    'xjxTXb',
    'RMdMXb',
    'arzKXb',
    '931RXb',
    'JnGKXb',
    'ihfMXb',
    'vfFRXb',
    'PUXTXb',
    'BnkOXb',
    'UeIWXb',
    'UnMWXb',
    'e9KWXb',
    'wJJWXb',
    'XSkYXb',
    'R4kYXb',
    '6MrZXb',
    '8KuZXb',
    'QTsZXb',
    'cWqZXb',
    ]
    for d in datasets:
        for run in baselines:
            print(f'Loading: {d} {run}', end=': ')
            runname = comments[run][0]
            runroot = os.path.join(runs_root, runname)
            net = runname.split('-')[0]
            if net == 'RecognitionNet':
                print('RecognitionNet')
                serfiq_path = os.path.join(runroot, f'quality-serfiq-{runname}-{d}.pickle')
                if os.path.isfile(serfiq_path):
                    print('Found serfiq')
                    with open(serfiq_path, 'rb') as f:
                        qualities[d+'+'+run+'+'+'serfiq'] = pickle.load(f)
                qualities[d+'+'+run+'+'+'magnitude'] = get_magnitude(run, runs_root, d)
            elif net == 'SddFiqaNet' or net == 'CrFiqaNet' or net == 'DfsNet':
                print('QualityNet')
                quality_path = os.path.join(runroot, f'quality-{runname}-{d}.pickle')
                if os.path.isfile(quality_path):
                    print('Found quality')
                    with open(quality_path, 'rb') as f:
                        qualities[d+'+'+run] = pickle.load(f)
                    for k in qualities[d+'+'+run]:
                        qualities[d+'+'+run][k] = np.average(qualities[d+'+'+run][k])
                        if net == 'DfsNet':
                            qualities[d+'+'+run][k] = -qualities[d+'+'+run][k]
            elif net == 'PfeNet':
                print('PfeNet')
                pfe_path = os.path.join(runroot, f'deviation-{runname}-{d}.pickle')
                if os.path.isfile(pfe_path):
                    print('Found pfe')
                    with open(pfe_path, 'rb') as f:
                        deviation = pickle.load(f)
                    qualities[d+'+'+run] = {}
                    for k in deviation:
                        qualities[d+'+'+run][k] = pfe_quality(np.exp(deviation[k]))
                        qualities[d+'+'+run][k] = -qualities[d+'+'+run][k]

    with open(quality_path, 'wb') as f:
        pickle.dump(qualities, f)



if __name__ == '__main__':
    import sys
    main(sys.argv[1])