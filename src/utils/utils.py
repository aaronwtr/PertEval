import warnings
import os
import requests

from tqdm import tqdm
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple
from zipfile import ZipFile

from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)

    if cfg.extras.get('distributed_storage'):
        # writing data_dir to cache for easy access
        if not os.path.exists(f'{os.getcwd()}/cache'):
            os.makedirs(f'{os.getcwd()}/cache')
        with open(f'{os.getcwd()}/cache/data_dir_cache.txt', 'w') as f:
            f.write(cfg.get("data")['data_dir'])


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def dataverse_download(url: str = "", save_path: str = ""):
    """Dataverse download helper with progress bar

    :param url: the url of the dataset
    :param save_path: the path to save the dataset
    """
    if os.path.exists(save_path):
        print('Found local copy...')
    else:
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        dir_path = save_path.split('/')[:-1]
        os.makedirs('/'.join(dir_path), exist_ok=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()


def zip_data_download_wrapper(url: str = "", zip_path: str = ""):
    """Wrapper for zip file download

    :param url: The url of the dataset.
    :param zip_path: The path where the file is downloaded.
    """
    if os.path.exists(zip_path):
        print('Found local copy of .zip file...')
    else:
        dataverse_download(url, zip_path + '.zip')
        print('Extracting zip file...')
        with ZipFile((zip_path + '.zip'), 'r') as z:
            z.extractall(path=os.path.dirname(zip_path))
        os.remove(zip_path + '.zip')
        print("Done!")


def find_root_dir(current_dir):
    while True:
        if '.project-root' in os.listdir(current_dir):
            return current_dir
        else:
            current_dir = os.path.dirname(current_dir)

        if current_dir == '/':  # if we have reached the root of the filesystem
            raise FileNotFoundError("Could not find .project-root file.")
