from . import PROCESSING_CONFIG_FILEPATH, MODEL_CONFIG_FILEPATH
from .dataloader import DataLoader 
from .logger import get_logger
from .idf import IDF
from .query_statistic import Statistic
from nltk.corpus import stopwords
import json
import math
from collections import Counter
from functools import reduce
logger = get_logger(__name__)

class Score:

    def __init__(self):
        with open(PROCESSING_CONFIG_FILEPATH, "r") as config_file:
            self._settings = json.load(config_file)
        with open(MODEL_CONFIG_FILEPATH, "r") as config_file:
            self._settings.update(json.load(config_file))
        self.stopwords = stopwords.words("russian")
        self.stopwords += [
            "купить", "2019", "год", "класс", "скачать", "отзыв",
            "работа", "москва", "песня", "сайт", "онлайн", "цена",
            "смотреть", "бесплатно", "русский", "игра"]

        self._dataloader = DataLoader()
        single_model_type = eval(self._settings["active_model"]["single"])
        single_model_args = self._settings["parametrs"]["single"]["model"][self._settings["active_model"]["single"]]["active"]
        self.single_model = single_model_type(**single_model_args)
        self.single_model_weight = self._settings["parametrs"]["single"]["weight"]
        
        self.pair_model_weight = self._settings["parametrs"]["pair"]["weight"]
        self.pair_model = Pair()

        self.all_words_model_weight = self._settings["parametrs"]["all_words"]["weight"]
        self.all_words_model = AllWords()

    def __call__(self, query_id, doc_id, relevant_doc_cnt):
        query = [word for word in self._dataloader.queries[query_id] if word not in self.stopwords]

        doc = self._dataloader.get_processed_file(doc_id)
        # doc["body"] = [[word for word in doc["body"] if word not in self.stopwords]]

        single_score = self.single_model_weight * self.single_model(query, doc, relevant_doc_cnt, query_id)
        pair_score = self.pair_model_weight * self.pair_model(query, doc)
        all_words_score = self.all_words_model_weight * self.all_words_model(query, doc)

        # print(single_score, pair_score, all_words_score)
        return single_score + pair_score + all_words_score

class SingleModel:
 
    def __init__(self):
        with open(MODEL_CONFIG_FILEPATH, "r") as config_file:
            self._settings= json.load(config_file)
        self.zones = self._settings["parametrs"]["single"]["zones_weight"] 


class tf_idf(SingleModel):

    def __init__(self, tf_type, df_type, normalization):
        super().__init__()
        self._df = IDF().load()
        self.tf = eval("self.tf_" + tf_type)
        self.df = eval("self._df." + df_type)
        self.normaliztion = eval("self.norm_" + normalization)
    
    def tf_natural(self, term, doc):
        return doc[term]

    def tf_logarithm(self, term, doc):
        return 1 + math.log(doc[term])

    def tf_augmented(self, term, doc):
        return 0.5 + 0.5 * doc[term] / doc.most_common(1)[0][1]
    
    def tf_log_ave(self, term, doc):
        avg_tf = sum(doc.values()) / len(doc)
        return (1 + math.log(doc[term])) / (1 + math.log(avg_tf))
        
    def norm_byte_size(self, body_counter, alpha=0.5):
        return 1 / sum([len(x[0]) * x[1] for x in body_counter.items()]) ** alpha
   
    def __call__(self, query, doc):
        title = doc["<title>"][0] if doc["<title>"] else []
        doc_counter = Counter(doc["body"][0] + title)
        if not doc_counter:
            return 0
        score = 0
        for zone in self.zones.values():
            tags = zone["tags"]
            weight = zone["weight"]
            for tag in tags:
                text = sum(doc[tag], [])
                for term in query:
                    if term in text and doc_counter[term]:
                        score += weight * self.tf(term, doc_counter) * self.df(term)
        return score * self.normaliztion(doc_counter)


class bm25yandex(SingleModel):
    
    def __init__(self, k_1, k_2):
        super().__init__()
        self._df = IDF().load()
        self.df = self._df.bm25 
        self.k_1 = k_1
        self.k_2 = k_2
    
    def __call__(self, query, doc):
        title = doc["<title>"][0] if doc["<title>"] else []
        doc_text = doc["body"][0] + title
        doc_counter = Counter(doc_text)
        if not doc_counter:
            return 0
        doc_len = len(doc_text)
        score = 0
        for zone in self.zones.values():
            tags = zone["tags"]
            weight = zone["weight"]
            for tag in tags:
                text = sum(doc[tag], [])
                for term in query:
                    if term in text and doc_counter[term]:
                        score += (self.tf1(term, doc_counter, doc_len) + 0.2 * weight) * self.df(term)
        return score
    
    def tf1(self, term, doc, doc_len):
        tf = doc[term]
        return tf / (tf + self.k_1 + doc_len / self.k_2)


class bm25sample(SingleModel):
    def __init__(self, k_1, b, avgdl):
        super().__init__()
        self._df = IDF().load()
        self.df = self._df.smoth_idf 
        self.k_1 = k_1
        self.b = b
        self.avgdl = avgdl
    
    def __call__(self, query, doc):
        title = doc["<title>"][0] if doc["<title>"] else []
        body = doc["body"][0]
        doc_text = body + title
        if not doc_text:
            return 0
        doc_counter = Counter(doc_text)
        doc_len = len(doc_text)
        score = 0
        for zone in self.zones.values():
            tags = zone["tags"]
            weight = zone["weight"]
            for tag in tags:
                text = sum(doc[tag], [])
                for term in query:
                    if term in text:
                        score += self.tf1(term, doc_counter, doc_len, weight) * self.df(term)
        return score
    
    def tf1(self, term, doc, doc_len, weight):
        tf = doc[term] * weight
        return tf * (self.k_1 + 1) / (tf + self.k_1 * (1 - self.b + self.b * doc_len / self.avgdl))

class bm25accurate(SingleModel):
    def __init__(self, k_1, b, k_3, avgdl):
        super().__init__()
        self._df = IDF().load()
        self.df = self._df.df
        self.statistic = Statistic().load()
        self.k_1 = k_1
        self.k_3 = k_3
        self.b = b
        self.avgdl = avgdl
        self.N = DataLoader().processed_content_len
    
    def __call__(self, query, doc, relevant_doc_cnt, query_id):
        title = doc["<title>"][0] if doc["<title>"] else []
        body = doc["body"][0]
        doc_text = body + title
        if not doc_text:
            return 0
        doc_counter = Counter(doc_text)
        doc_len = len(doc_text)
        N = self.N       
        k_1 = self.k_1
        b = self.b
        k_3 = self.k_3
        avgdl = self.avgdl
        VR = relevant_doc_cnt
        
        score = 0
        for term in query:
            term_weight = 0
            for zone in self.zones.values():
                tags = zone["tags"]
                zone_weight = zone["weight"]
                text = set()
                for tag in tags:
                    text = text | set(sum(doc[tag], []))
                if term in text:
                    term_weight += zone_weight 
                
            VRt, VNRt = self.statistic(query_id)[term]
            df = self.df(term)
            tf = doc_counter.get(term, 0) * term_weight
            # tf = doc_counter.get(term, 0)


            A = ((VRt + 0.5) / (VNRt + 0.5)) / ((df - VRt + 0.5) / (N - df - VR + VRt + 0.5))
            B = tf * (k_1 + 1) / (tf + k_1 * (1 - b + b * doc_len / avgdl))
            C = (k_3 + 1) * tf / (k_3 + tf)
            #if A < 1:
            #    print("term:", term, "VRt: ", VRt, ";VNRt: ", VNRt, ";df: ", df, ";N: ", N,";VR: ", VR)

            if A * B * C > 1:
                score += math.log(A * B * C)
        return score

class Pair:
    
    def __init__(self):
        self._df = IDF().load()
        self.df = self._df.prob_idf
    
    def __call__(self, query, doc):
        title = doc["<title>"][0] if doc["<title>"] else []
        body = doc["body"][0]
        doc_text = body + title
        p = [self.df(word) for word in query]
        query_last = len(query) - 2
        doc_last = len(doc) - 3
        score = 0
        for query_start in range(query_last):
            query_tuple = query[query_start:query_start + 2]
            tf = 0
            for doc_start in range(doc_last):
                word_tuple = doc_text[doc_start: doc_start + 2]
                word_over_tuple = doc_text[doc_start: doc_start + 3: 2]
                if query_tuple == word_tuple:
                    tf += 1
                if query_tuple == word_tuple[::-1]:
                    tf += 0.5
                if query_tuple == word_over_tuple:
                    tf += 0.5
            score += 0.3 * (p[query_start] + p[query_start + 1]) * tf / (1 + tf)
        return score


class AllWords:

    def __init__(self):
        self._df = IDF().load()
        self.df = self._df.prob_idf

    def __call__(self, query, doc):
        title = doc["<title>"][0] if doc["<title>"] else []
        body = doc["body"][0]
        words = set(query) - set(body + title)
        score = 0.2 * sum([self.df(word) for word in query])
        score *= 0.03 ** len(words)
        return score
