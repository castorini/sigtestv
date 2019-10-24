from dataclasses import dataclass
import json

import cherrypy
import requests

from sigtestv.evaluate import PipelineComponent, ExperimentResult


@dataclass
class NetLogger(PipelineComponent):
    endpoint: str

    def __call__(self, run):
        run_config = run.run_config
        results = run.results
        data = [run_config.model_name,
                run_config.dataset_name,
                run_config.command_base,
                [x.tolist() for x in results],
                list(run_config.hyperparameters.items())]
        try:
            requests.post(self.endpoint, json=data, timeout=10)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass


@dataclass
class OfflineNetLogger(PipelineComponent):
    filename: str

    def __call__(self, run):
        run_config = run.run_config
        results = run.results
        data = [run_config.model_name,
                run_config.dataset_name,
                run_config.command_base,
                [x.tolist() for x in results],
                list(run_config.hyperparameters.items())]
        with open(self.filename, 'a') as f:
            print(json.dumps(data), file=f)


def replay_data_log(filename, endpoint):
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        requests.post(endpoint, json=data)


@cherrypy.expose
class DatabaseLoggingService(object):

    def __init__(self, database):
        self.database = database

    @cherrypy.tools.json_in()
    def POST(self, **kwargs):
        data_json = cherrypy.request.json
        results = [ExperimentResult.fromlist(x) for x in data_json[3]]
        model_name, dataset_name, command_base, _, hp_items = data_json
        self.database.insert_result(model_name, dataset_name, command_base, results, hp_items)
