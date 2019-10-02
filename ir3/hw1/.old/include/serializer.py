# coding: utf-8

import msgpack
import logging.config, logging

logging.config.fileConfig(fname='logging.conf', disable_existing_loggers=False)
logger = logging.getLogger('logger.serializer')

# Save data to filepath via messagepack serialization
def save(data, filepath):
    try:
        with open(filepath, 'wb') as out_file:
            msgpack.pack(data, out_file)
    except Exception as e:
        logger.exception('Serialization error occured')
        return None
    else:
        logger.info('Serialization data: OK, saved to file {}'.format(filepath))
        return True

# Load data from filepath
def load(filepath, use_list=True):
    data = None
    try:
        with open(filepath, 'rb') as in_file:
            data = msgpack.unpack(in_file, use_list=use_list, raw=False)
    except Exception as e:
        logger.exception('Deserialization error occured')
        return None
    else:
        return data

'''
d = {1:'asf',
     2:'dsfasdf',
     3:'sdfs'}

save(d, './tmp.dat')
d_ = load('./tmp.dat')
print(d, d_)
predicate = d_ == d
print(predicate * 'OK' + (not predicate) * 'Error')
'''
