import os
import sys
import math
import json
import torch
import random
import logging
import numpy as np


def init_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    
def save_model(model, score_func, optimizer, save_path):
    """
    Save model
    """
    if not os.path.isdir(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    print(f"Saving model to {save_path}...")

    state = {
        'model'	: model.state_dict(),
        "score_func": score_func.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, save_path)


def load_model(model, score_func, checkpoint, device):
    """
    Load saved models
    """
    state = torch.load(checkpoint, map_location="cpu")
    keys = model.load_state_dict(state["model"]) #, strict=False)
    # print("Model Unmatched Params", keys, flush=True)
    keys = score_func.load_state_dict(state['score_func']) #, strict=False)
    # print("Model Unmatched Params", keys, flush=True)

    model = model.to(device)
    score_func = score_func.to(device)

    return model, score_func


def get_logger(name, log_dir, config_dir):	
    """
	Creates a logger object
	Parameters
	----------
	name:           Name of the logger file
	log_dir:        Directory where logger file needs to be stored
	config_dir:     Directory from where log_config.json needs to be read
	
	Returns
	-------
	A logger object which writes to both file and stdout
	"""
    config_dict = json.load(open( config_dir + '/log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)


    def get_best_epochs(self, eval_steps, run=None):
        # Return the epoch with the best val performance for each seed
        best_results = []

        for r in self.results:
            r = torch.tensor(r)
            best_val_epoch = eval_steps * (r[:, 1].argmax() + 1)
            best_results.append(best_val_epoch)
        
        return best_results


    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            best_results = []

            for r in self.results:
                r = 100 * torch.tensor(r)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            # print(f'All runs:')

            r = best_result[:, 0]
            # print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')

            r = best_result[:, 1]
            best_valid_mean = round(r.mean().item(), 2)
            best_valid_var = round(r.std().item(), 2)

            best_valid = str(best_valid_mean) +' ' + '±' +  ' ' + str(best_valid_var)
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')

            r = best_result[:, 2]
            best_train_mean = round(r.mean().item(), 2)
            best_train_var = round(r.std().item(), 2)
            # print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')

            r = best_result[:, 3]
            best_test_mean = round(r.mean().item(), 2)
            best_test_var = round(r.std().item(), 2)
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            mean_list = [best_train_mean, best_valid_mean, best_test_mean]
            var_list = [best_train_var, best_valid_var, best_test_var]

            # return best_valid, best_valid_mean, mean_list, var_list
            return mean_list, var_list



def rename_best_saved(logger, model_save_name, eval_steps, rand_split=False):
    """
    Rename the file name for the best performing model

    We basically just remove the epoch info
    """
    num_runs = len(logger.results)
    runtype = "split" if rand_split else "seed"

    for run, epoch in enumerate(logger.get_best_epochs(eval_steps)):
        if num_runs > 1:
            existing_name = f"{model_save_name}_{runtype}-{run+1}_epoch-{epoch}.pt"
            new_name = f"{model_save_name}_{runtype}-{run+1}.pt"
        else:
            existing_name = f"{model_save_name}_epoch-{epoch}.pt"
            new_name = f"{model_save_name}.pt"

        os.rename(existing_name, new_name)


