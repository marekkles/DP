import pickle
import torch
import os

import serfiq
import dataset
from tqdm import tqdm

def main(model, datamodule, device, run_root, run_name):
    for predict_ds in datamodule.iris_predict_list:
        print(f"Predict for dataset: {predict_ds}")
        q = {}
        ds = datamodule.iris_predict[predict_ds]
        for image, label in tqdm(ds, "Predicting"):
            quality = serfiq.quality_score(model, image, device)
            q[label] = quality
        with open(os.path.join(
            run_root, f'quality-serfiq-{run_name}-{predict_ds}.pickle'
        ), 'wb') as f:
            pickle.dump(q, f)

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="ArcFace model evaluation :get: embeddings", add_help=add_help)
    parser.add_argument("--run-root", type=str, help="Run root")
    parser.add_argument("--dataset-root", type=str, help="Dataset root")
    parser.add_argument("--device", type=str, help="Device to use")
    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    print("Loading model")
    model, args_file = serfiq.load_model(args.run_root)
    print("Loading dataset")
    datamodule = dataset.IrisDataModule(
        args.dataset_root, 1, 1, 1,
        train_pseudolabels= None,
        train_subset= args_file['dataset_subsets'],
        auto_crop= args_file['auto_crop'],
        unwrap= args_file['unwrap'],
        shuffle= args_file['shuffle'],
        train_transform= dataset.train_transform(**args_file['train_transform']),
        val_transform= dataset.val_transform(**args_file['val_transform']),
        test_transform= dataset.test_transform(**args_file['test_transform']),
        predict_transform= dataset.predict_transform(**args_file['predict_transform'])
    )
    print("Setting up model")
    serfiq.setup_model(model)
    print(f"Sending model to device {args.device}")
    device = torch.device(args.device)
    model = model.to(device)
    main(model, datamodule, device, args.run_root, args_file['run_name'])
    

