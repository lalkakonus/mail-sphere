from include.dataloader import DataLoader 
from include.score import Score
from include.logger import get_logger
from progressbar import progressbar
import math
import json
PROCESSING_CONFIG_FILEPATH = "config/processing.config"
logger = get_logger(__name__)

class Ranger:

    def __init__(self):
        with open(PROCESSING_CONFIG_FILEPATH, "r") as config_file:
            self._settings = json.load(config_file)
        self._dataloader = DataLoader()
        self._result = {}
        self._score = Score()
    
    def predict(self):
        logger.info("Prediction started.")
        bar = progressbar(range(len(self._dataloader.queries)))
        for query_id, doc_ids in self._dataloader.submission.items():
            ranged_list = [[self._score(query_id, doc_id), doc_id] for doc_id in doc_ids]
            ranged_list.sort(key=lambda x: x[0], reverse=True)
            ranged_list = [x[1] for x in ranged_list]
            self._result[query_id] = ranged_list
            bar.__next__()
        logger.info("Prediction finished.")
        return self

    def save(self):
        with open(self._settings["filepath"]["submission"], "w") as submission_file:
            submission_file.write("QueryId,DocumentId\n")
            for query_id, doc_ids in self._result.items():
                for doc_id in doc_ids:
                    submission_file.write(str(query_id) + "," + str(doc_id) + "\n")
        logger.info("Prediction saved to '{}'.".format(self._settings["filepath"]["submission"]))

if __name__ == "__main__":
    ranger = Ranger().predict()
    ranger.save()
