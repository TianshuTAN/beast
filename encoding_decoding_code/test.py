import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analyses.utils.utils import (
    set_seed,
    get_args,
)
from analyses.utils.encoder import (
    train_rrr_encoder_with_tune,
    train_cnn_encoder_with_tune,
)
from analyses.utils.decoder import (
    train_rrr_decoder_with_tune,
    train_cnn_decoder_with_tune,
)
import copy
import numpy as np


def main():
    args = get_args()

    eid = args.eid
    neural_input_dir = os.path.join(args.neural_input_dir, eid)
    latent_input_dir = os.path.join(args.latent_input_dir, eid)
    seed = args.seed
    eval_task = args.eval_task

    if eval_task == 'encoding':
        print("Neural Encoding:")
        save_path = os.path.join(latent_input_dir, 'encoding_results')
    elif eval_task == 'decoding':
        print("Neural Decoding:")
        save_path = os.path.join(latent_input_dir, 'decoding_results')
    else:
        raise ValueError(f"Invalid evaluation task: {eval_task}")

    # set seed
    set_seed(seed)
    
    result_dict = {}

    print(f"Processing {eid}")

    # load data (.npz is dict-like; do not use .items() — that returns a view, not a mapping)
    neural_data_dict = np.load(
        os.path.join(neural_input_dir, f"{eid}_aligned.npz"), allow_pickle=True
    )
    latent_data_dict = np.load(
        os.path.join(latent_input_dir, "z_trials.npz"), allow_pickle=True
    )

    all_embeddings = latent_data_dict['z_trials_time']
    trial_split = latent_data_dict['trial_split']

    # Flatten embeddings across camera view directions
    K, T, V, D = all_embeddings.shape
    all_embeddings = all_embeddings.reshape(K, T, V * D)

    len_train = len(neural_data_dict['train_intervals'])
    len_val = len(neural_data_dict['val_intervals'])
    len_test = len(neural_data_dict['test_intervals'])

    # CAUTION: check whether the order of the neural data and latent embeddings is correct!
    train_neural = neural_data_dict['train_spikes']
    val_neural = neural_data_dict['val_spikes']
    test_neural = neural_data_dict['test_spikes']
    train_embedding = all_embeddings[:len_train]
    val_embedding = all_embeddings[len_train:len_train+len_val]
    test_embedding = all_embeddings[len_train+len_val:]
    
    train_data = { eid: { "X": [], "y": [], "setup": {} } }

    if eval_task == 'encoding':
        train_data[eid]["X"].append(train_embedding)
        train_data[eid]["X"].append(val_embedding)
        train_data[eid]["X"].append(test_embedding)
        train_data[eid]["y"].append(train_neural)
        train_data[eid]["y"].append(val_neural)
        train_data[eid]["y"].append(test_neural)
    elif eval_task == 'decoding':
        train_data[eid]["X"].append(train_neural)
        train_data[eid]["X"].append(val_neural)
        train_data[eid]["X"].append(test_neural)
        train_data[eid]["y"].append(train_embedding)
        train_data[eid]["y"].append(val_embedding)
        train_data[eid]["y"].append(test_embedding)

    print(
        f"Train X Shape: {train_data[eid]['X'][0].shape}, Train Y Shape: {train_data[eid]['y'][0].shape}"
    )
    print(
        f"Val X Shape: {train_data[eid]['X'][1].shape}, Val Y Shape: {train_data[eid]['y'][1].shape}"
    )
    print(
        f"Test X Shape: {train_data[eid]['X'][2].shape}, Test Y Shape: {train_data[eid]['y'][2].shape}"
    )
    train_data_cnn = copy.deepcopy(train_data)

    num_samples = int(os.environ.get("NUM_SAMPLES", 20))
    print(f"Using num_samples={num_samples} for Ray Tune")

    if eval_task == 'encoding':
        rrr_result = train_rrr_encoder_with_tune(train_data, num_samples=num_samples)
        cnn_result = train_cnn_encoder_with_tune(train_data, num_samples=num_samples)
    elif eval_task == 'decoding':
        rrr_result = train_rrr_decoder_with_tune(train_data, num_samples=num_samples)
        cnn_result = train_cnn_decoder_with_tune(train_data, num_samples=num_samples)
    else:
        raise ValueError(f"Invalid evaluation task: {eval_task}")
    
    if eval_task == 'encoding':
        print(
            f"RRR Encoding {eid} Test BPS: {rrr_result[eid]['bps']} Test R2: {rrr_result[eid]['r2']}"
        )
        print(
            f"CNN Encoding {eid} Test BPS: {cnn_result[eid]['bps']} Test R2: {cnn_result[eid]['r2']}"
        )
    elif eval_task == 'decoding':
        print(f"RRR Decoding {eid} Test R2: {rrr_result[eid]['r2']}")
        print(f"CNN Decoding {eid} Test R2: {cnn_result[eid]['r2']}")
    
    result_dict[eid] = {
        'rrr': rrr_result[eid],
        'cnn': cnn_result[eid]
    }
    
    # save result dict
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path, result_dict)


if __name__ == '__main__':
    main()
