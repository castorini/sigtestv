from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple
import sqlite3

from sigtestv.evaluate import ExperimentResult, RunConfiguration, PipelineComponent, RunCollection, CompletedRun


class ConnectionContextManager(object):

    def __init__(self, filename: str):
        self.filename = filename
        self.opened = 0

    def __enter__(self):
        if self.opened == 0:
            self.connection = sqlite3.connect(self.filename)
            self.cursor = self.connection.cursor()
        self.opened += 1
        return self

    def __exit__(self, *args):
        self.opened -= 1
        if self.opened == 0:
            self.connection.commit()
            self.connection.close()


def open_context(filename):
    return ConnectionContextManager(filename)


class ResultsDatabase(object):

    def __init__(self, filename):
        self.filename = filename
        with open_context(filename) as ctx:
            c = ctx.cursor
            c.execute('''CREATE TABLE IF NOT EXISTS model(id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)''')
            c.execute('''CREATE INDEX IF NOT EXISTS model_name_i ON model(name)''')
            c.execute('''CREATE TABLE IF NOT EXISTS experiment(id INTEGER PRIMARY KEY, model_id INTEGER NOT NULL,
                         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, command_text TEXT NOT NULL, dataset_name TEXT NOT NULL,
                         FOREIGN KEY(model_id) REFERENCES model(id) ON DELETE CASCADE)''')
            c.execute('''CREATE INDEX IF NOT EXISTS experiment_model_id_fi ON experiment(model_id)''')
            c.execute('''CREATE INDEX IF NOT EXISTS experiment_dataset_i ON experiment(dataset_name)''')
            c.execute('''CREATE TABLE IF NOT EXISTS experiment_result_assoc(id INTEGER PRIMARY KEY, exp_id INTEGER NOT NULL,
                         metric_name TEXT NOT NULL, result REAL NOT NULL, set_type TEXT NOT NULL, FOREIGN KEY(exp_id) REFERENCES experiment(id) ON DELETE CASCADE)''')
            c.execute('''CREATE INDEX IF NOT EXISTS experiment_result_assoc_exp_id_fi ON experiment_result_assoc(exp_id)''')
            c.execute('''CREATE TABLE IF NOT EXISTS experiment_hyperparameter_assoc(id INTEGER PRIMARY KEY, exp_id INTEGER NOT NULL,
                         name TEXT NOT NULL, value TEXT, FOREIGN KEY(exp_id) REFERENCES experiment(id) ON DELETE CASCADE)''')
            c.execute('''CREATE INDEX IF NOT EXISTS experiment_hyperparameter_assoc_exp_id_fi ON experiment_hyperparameter_assoc(exp_id)''')

    def fetch_all(self, model_name, dataset_name):
        query = '''SELECT * FROM MODEL JOIN experiment ON model.id=model_id JOIN experiment_result_assoc ON 
                   experiment_result_assoc.exp_id=experiment.id JOIN experiment_hyperparameter_assoc ON 
                   experiment.id=experiment_hyperparameter_assoc.exp_id WHERE model.name=? AND dataset_name=?'''
        data = [model_name, dataset_name]
        exp_metadata = {}
        exp_opts = defaultdict(list)
        exp_results = defaultdict(list)
        with open_context(self.filename) as ctx:
            c = ctx.cursor
            c.execute(query, data)
            for row in c.fetchall():
                model_id, model_name, exp_id, model_id, timestamp, cmd_txt, ds_name, exp_id, exp_id, measure_name, \
                    result, set_type, _, _, opt_name, opt_value = row
                exp_results[exp_id].append((result, measure_name, set_type))
                exp_opts[exp_id].append((opt_name, opt_value))
                exp_metadata[exp_id] = (model_name, cmd_txt, ds_name)

        configs = []
        results = []
        metadatas = []
        for exp_id, (model_name, cmd_txt, ds_name) in exp_metadata.items():
            options = dict(filter(lambda x: x[0].startswith('-'), set(exp_opts[exp_id])))
            env_vars = dict(filter(lambda x: not x[0].startswith('-'), set(exp_opts[exp_id])))
            configs.append(RunConfiguration(model_name, cmd_txt, ds_name, options, env_vars))
            results.append([ExperimentResult(*x) for x in set(exp_results[exp_id])])
            metadatas.append(dict(exp_id=exp_id))
        runs = [CompletedRun(cfg, res, metadata=md) for cfg, res, md in zip(configs, results, metadatas)]
        return RunCollection(runs)

    def add_result(self, exp_id, results: List[ExperimentResult], ctx=None):
        if ctx is None: ctx = open_context(self.filename)
        with ctx:
            c = ctx.cursor
            query = 'INSERT INTO experiment_result_assoc(exp_id, metric_name, result, set_type) VALUES (?, ?, ?, ?)'
            values = []
            for result in results:
                values.append((exp_id, result.name, result.value, result.set_type))
            for value in values:
                c.execute(query, value)

    def insert_result(self,
                      model_name: str,
                      dataset_name: str,
                      full_command_str: str,
                      results: List[ExperimentResult],
                      hyperparameters: List[Tuple[str, str]]):
        with open_context(self.filename) as ctx:
            c = ctx.cursor
            c.execute('SELECT id FROM model WHERE name=?', (model_name,))
            row = c.fetchone()
            if row:
                model_id = row[0]
            else:
                c.execute('INSERT INTO model(name) VALUES (?)', (model_name,))
                model_id = c.lastrowid

            c.execute('INSERT INTO experiment(model_id, command_text, dataset_name) VALUES (?, ?, ?)', (model_id, full_command_str, dataset_name))
            exp_id = c.lastrowid
            self.add_result(exp_id, results, ctx=ctx)

            values = [(exp_id, x[0], x[1]) for x in hyperparameters]
            query = f'INSERT INTO experiment_hyperparameter_assoc(exp_id, name, value) VALUES (?, ?, ?)'
            for value in values:
                c.execute(query, value)


@dataclass
class DatabaseLogger(PipelineComponent):
    database: ResultsDatabase
    log_env_vars: bool = False

    def __call__(self, run: CompletedRun):
        run_config = run.run_config
        results = run.results
        log_options = list(run_config.hyperparameters.items()) if self.log_env_vars else list(run_config.options)
        self.database.insert_result(run_config.model_name,
                                    run_config.dataset_name,
                                    run_config.command_base,
                                    results,
                                    log_options)


@dataclass
class DatabaseUpdateLogger(PipelineComponent):
    database: ResultsDatabase

    def __call__(self, run):
        self.database.add_result(run.metadata['exp_id'], run.results)
