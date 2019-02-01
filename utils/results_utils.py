import os
import yaml

from collections import defaultdict

import utils.file_utils as file_utils


class ExperimentResults(object):
    def __init__(self, results_dir):

        self.results_dir = results_dir
        self.results = []
        self.config = {}

        self.name = os.path.split(results_dir)[-1]

        self.load_results()
        self.load_config()

    def load_results(self):
        self.results = file_utils.get_results_from_dir(self.results_dir)

    def load_config(self):
        if os.path.exists((os.path.join(self.results_dir, 'config.yaml'))):
            self.config = yaml.load(open(os.path.join(self.results_dir, 'config.yaml'), 'rb'))
        else:
            files = file_utils.get_immediate_files(self.results_dir)

            for file in files:
                if file.split('.')[-1] == 'yaml':
                    self.config = yaml.load(open(os.path.join(self.results_dir, file), 'rb'))

    def filter_results(self, filter):
        return [r for r in self.results if filter(r)]

    def group_results(self, group_function):
        ret = defaultdict(list)

        for r in self.results:
            ret[group_function(r)].append(r)

        return ret
