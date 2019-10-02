# coding: utf-8
import msgpack
import os
from .logger import get_logger
logger = get_logger(__name__)

class Serializer:

    @staticmethod
    def save(data, filepath, force=False):
        answer = ''
        if force and os.path.exist(filepath):
            while not answer.lower() in {"y", "n"}:
                answer = input("File '{}' already exist, do you want to replace it? [Y/N] ".format(filepath))
                if answer == 'n':
                    return None
        try:
            with open(filepath, 'wb') as out_file:
                msgpack.pack(data, out_file)
        except Exception as error:
            logger.exception('Serialization error occured: {}'.format(str(error)))
            return None
        finally:
            if type(filepath) == os.DirEntry:
                filepath = filepath.name
            logger.debug("Serialization data complete and saved to '{}'".format(filepath))
            return data

    @staticmethod
    def load(filepath, use_list=True):
        data = None
        try:
            with open(filepath, 'rb') as in_file:
                data = msgpack.unpack(in_file, use_list=use_list, raw=False)
        except Exception as error:
            logger.exception('Deserialization error occured: {}'.format(str(error)))
            return None
        finally:
            if type(filepath) == os.DirEntry:
                filepath = filepath.name
            logger.debug("Deserialization data from '{}' complete.".format(filepath))
            return data
