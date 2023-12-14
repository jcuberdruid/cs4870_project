# File: projectDir/src/utilities/experiments.py
import os
import json
import shutil
import uuid
import sys
from datetime import datetime


# Hacky import because python sucks
script_dir = os.path.dirname(__file__)  # gets the directory of the current file
parent_dir = os.path.dirname(script_dir)  # gets the parent directory (project/src)
sys.path.append(parent_dir)

import paths

class ExperimentManager:
    def __init__(self):
        self.experiment_dir = paths.experimentDir
        self.results_dir = paths.resultDir
        self.archive_dir = paths.archiveDir

    def new_experiment(self, name, description):
        uid = self.generateUID()
        print(uid)
        exp_name_uid = f"{uid}"
        os.makedirs(os.path.join(self.results_dir, ("results_"+exp_name_uid)))
        os.makedirs(os.path.join(self.experiment_dir, exp_name_uid))

        info = {
            "name": name,
            "description": description,
            "ID": uid
        }
        with open(os.path.join(self.experiment_dir, exp_name_uid, "info.json"), 'w') as f:
            json.dump(info, f)

    def generateUID(self):
        while True:
            uid = str(uuid.uuid4())
            if not self._is_duplicate_uid(uid):
                return uid

    def _is_duplicate_uid(self, uid):
        for directory in [self.experiment_dir, self.archive_dir]:
            if uid in os.listdir(directory):
                return True
        return False

    def archive(self, ID):
        exp_path = os.path.join(self.experiment_dir, ID)
        results_path = os.path.join(self.results_dir, ("results_"+ID))
        archive_path = os.path.join(self.archive_dir, ID)

        if os.path.exists(exp_path) and os.path.exists(results_path):
            shutil.move(exp_path, os.path.join(archive_path, ID))
            shutil.move(results_path, archive_path)
            shutil.make_archive(archive_path, 'zip', archive_path)
        else: 
            print("rut roh: no path ")

    def unarchive(self, ID):
        archive_path = os.path.join(self.archive_dir, ID + '.zip')
        if os.path.exists(archive_path):
            shutil.unpack_archive(archive_path, self.archive_dir)
            shutil.move(os.path.join(self.archive_dir, ID, 'results'), self.results_dir)
            shutil.move(os.path.join(self.archive_dir, ID, 'experiments'), self.experiment_dir)

    def mostRecent(self):
        latest = None
        latest_time = None
        for exp in os.listdir(self.experiment_dir):
            exp_path = os.path.join(self.experiment_dir, exp)
            mtime = os.path.getmtime(exp_path)
            if latest is None or mtime > latest_time:
                latest = exp
                latest_time = mtime

        if latest:
            with open(os.path.join(self.experiment_dir, latest, "info.json")) as f:
                info = json.load(f)
            print(f"Name: {info['name']}\nID: {info['ID']}\nDescription: {info['description']}\nLast Modified: {datetime.fromtimestamp(latest_time)}")
            print(f"Path: {os.path.join(self.experiment_dir, latest)}")

# Example usage:
manager = ExperimentManager()
manager.new_experiment("Test", "Description of the experiment")
#manager.archive("bbfde84a-364f-4190-b0c3-77b41d1af75f")
