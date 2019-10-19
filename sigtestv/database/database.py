from dataclasses import dataclass
from typing import List, Tuple
import sqlite3

from sigtestv.evaluate import ExperimentResult, RunConfiguration, PipelineComponent


class ResultsDatabase(object):

    def __init__(self, filename):
        self.filename = filename
        conn = sqlite3.connect(filename)
        with conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS model(id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)''')
            c.execute('''CREATE INDEX IF NOT EXISTS model_name_i ON model(name)''')
            c.execute('''CREATE TABLE IF NOT EXISTS experiment(id INTEGER PRIMARY KEY, model_id INTEGER NOT NULL,
                         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, command_text TEXT NOT NULL, dataset_name TEXT NOT NULL,
                         FOREIGN KEY(model_id) REFERENCES model(id))''')
            c.execute('''CREATE INDEX IF NOT EXISTS experiment_model_id_fi ON experiment(model_id)''')
            c.execute('''CREATE INDEX IF NOT EXISTS experiment_dataset_i ON experiment(dataset_name)''')
            c.execute('''CREATE TABLE IF NOT EXISTS experiment_result_assoc(id INTEGER PRIMARY KEY, exp_id INTEGER NOT NULL,
                         metric_name TEXT NOT NULL, result REAL NOT NULL, set_type TEXT NOT NULL, FOREIGN KEY(exp_id) REFERENCES experiment(id))''')
            c.execute('''CREATE INDEX IF NOT EXISTS experiment_result_assoc_exp_id_fi ON experiment_result_assoc(exp_id)''')
            c.execute('''CREATE TABLE IF NOT EXISTS experiment_hyperparameter_assoc(id INTEGER PRIMARY KEY, exp_id INTEGER NOT NULL,
                         name TEXT NOT NULL, value TEXT, FOREIGN KEY(exp_id) REFERENCES experiment(id))''')
            c.execute('''CREATE INDEX IF NOT EXISTS experiment_hyperparameter_assoc_exp_id_fi ON experiment_hyperparameter_assoc(exp_id)''')
            conn.commit()

    def insert_result(self,
                      model_name: str,
                      dataset_name: str,
                      full_command_str: str,
                      results: List[ExperimentResult],
                      hyperparameters: List[Tuple[str, str]]):
        conn = sqlite3.connect(self.filename)
        with conn:
            c = conn.cursor()
            c.execute('SELECT id FROM model WHERE name=?', (model_name,))
            row = c.fetchone()
            if row:
                model_id = row[0]
            else:
                c.execute('INSERT INTO model(name) VALUES (?)', (model_name,))
                model_id = c.lastrowid

            c.execute('INSERT INTO experiment(model_id, command_text, dataset_name) VALUES(?, ?, ?)', (model_id, full_command_str, dataset_name))
            exp_id = c.lastrowid

            values = []
            for result in results:
                values.append((exp_id, result.name, result.value, result.set_type))
            query = f'INSERT INTO experiment_result_assoc(exp_id, metric_name, result, set_type) VALUES (?, ?, ?, ?)'
            for value in values:
                c.execute(query, value)

            values = [(exp_id, x[0], x[1]) for x in hyperparameters]
            query = f'INSERT INTO experiment_hyperparameter_assoc(exp_id, name, value) VALUES (?, ?, ?)'
            for value in values:
                c.execute(query, value)
            conn.commit()


@dataclass
class DatabaseLogger(PipelineComponent):
    database: ResultsDatabase

    def __call__(self, run_config: RunConfiguration, results: List[ExperimentResult]):
        self.database.insert_result(run_config.model_name,
                                    run_config.dataset_name,
                                    run_config.command_base,
                                    results,
                                    list(run_config.hyperparameters.items()))
