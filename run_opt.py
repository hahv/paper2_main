from tap import *
import optuna
from optuna.trial import Trial

from halib import *
from halib.system.path import *
from halib.exp.core.param_gen import ParamGen

from run_exp import run_single_exp


class RunOptArgs(Tap):
    cfg: str = r"./config/zruns/__base.yaml"
    optcfg: str = r"./config/zruns/__opt_cfg.yaml"

SEARCH_SPACE = None
current_exp_cfg_file = None


# Define evaluation function
def objective(trial: Trial):
    global SEARCH_SPACE
    num_trials = calc_num_trials(SEARCH_SPACE)

    trial_param_set = {}
    for params in SEARCH_SPACE:  # ty:ignore[not-iterable]
        print(f"{params}: {SEARCH_SPACE[params]}")  # ty:ignore[not-subscriptable]
        value = trial.suggest_categorical(params, SEARCH_SPACE[params])  # ty:ignore[not-subscriptable]
        trial_param_set[params] = value

    # ---- Run your pipeline with these hyperparams ----
    global current_exp_cfg_file
    with ConsoleLog(f"Running trial {trial.number}/{num_trials}"):
        print("param set :")
        pprint(trial_param_set)
        metrics = run_single_exp(current_exp_cfg_file, method_cfg_dict=trial_param_set)

    # Return metric to maximize (e.g., F1 score)
    return np.random.rand()


def calc_num_trials(search_space):
    count = 1
    for param in search_space:
        count *= len(search_space[param])
    return count


def main():
    # move to folder where the script is located
    script_folder = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_folder)
    args = RunOptArgs().parse_args()

    # Load base config for experiments
    global current_exp_cfg_file
    current_exp_cfg_file = args.cfg
    assert os.path.exists(current_exp_cfg_file), (
        f"Config file {current_exp_cfg_file} does not exist."
    )

    # Load optimization config
    opt_cfg_file = args.optcfg
    global SEARCH_SPACE
    SEARCH_SPACE = ParamGen.from_file(opt_cfg_file).params

    # Create a GridSampler with your space
    sampler = optuna.samplers.GridSampler(SEARCH_SPACE)

    # Persistent storage
    prefix = "sqlite:///"
    sqlite_db_path = os.path.abspath(os.path.join(
        script_folder, "zout/tune/optuna_study.db"
    ))
    sqlite_db_path = normalize_paths(sqlite_db_path)
    storage_url = prefix + sqlite_db_path
    study = optuna.create_study(
        study_name=f"temp_stabilize_opt_{now_str()}",
        direction="maximize",
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True,
    )
    num_trials = calc_num_trials(SEARCH_SPACE)
    console.rule(f"Total trials to run: {num_trials}")
    time.sleep(5)
    # Run optimization
    study.optimize(objective)

    print("Best params:", study.best_params)
    print("Best value:", study.best_value)


if __name__ == "__main__":
    main()
