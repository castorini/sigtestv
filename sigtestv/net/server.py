from dataclasses import dataclass

import cherrypy
import requests

from sigtestv.evaluate import PipelineComponent, ExperimentResult


@dataclass
class NetLogger(PipelineComponent):
    endpoint: str

    def __call__(self, run_config, results):
        data = [run_config.model_name,
                run_config.dataset_name,
                run_config.command_base,
                [x.tolist() for x in results],
                list(run_config.hyperparameters.items())]
        try:
            requests.post(self.endpoint, json=data)
        except requests.exceptions.ConnectionError:
            pass


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
