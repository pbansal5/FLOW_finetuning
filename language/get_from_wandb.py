import wandb
import numpy as np
import argparse

def get_average(project_name, runs, tasks, name):
    api = wandb.Api()
    for run_id in runs:
        vals = []
        run = api.run(f"{project_name}/{run_id}")
        summary = run.summary
        for task in tasks:
            vals.append(summary[task])
        avg = np.mean(vals)
        print ("run name : %s, Task : %s, Value : %0.3f"%(run.name,name, avg))
        summary[name] = avg
        summary.update()

project_name = "flow"
runs = [
     "27dzy324", 
]

get_average(
    project_name,
    runs,
    [
        "arc_easy/acc",
        "arc_challenge/acc",
        "hellaswag/acc",
        "piqa/acc",
        "social_iqa/acc",
        "openbookqa/acc",
    ],
    "commonsense/avg",
)
get_average(
    project_name,
    runs,
    ["commonsense/avg", "mmlu/acc", "mbpp/pass_at_1"],
    "Task_1/avg",
)
get_average(
    project_name,
    runs,
    ["commonsense/avg", "mmlu/acc", "mbpp/pass_at_1", "gsm8k/exact_match,strict-match"],
    "Average",
)
