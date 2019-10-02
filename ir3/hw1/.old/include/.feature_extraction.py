# coding: utf-8
import zlib, re, sys
sys.path.insert(0, 'include')
from tokenizer import pymorphy_tokenizer
from key_words import get_pop_words, get_distinct_words
from serializer import load
from bs4 import BeautifulSoup
from lang_model import LangaugeModel
from nltk.corpus import stopwords

popular_words, distinct_words = None, None 
langauge_model = LangaugeModel()
STOP_WORDS = set(stopwords.words('russian'))

def decode(words):
    tmp = []
    for word in words:
        tmp += word.decode('utf-8')
    return tmp

def upload_data():
    global popular_words, distinct_words
    popular_words, distinct_words = load('data/key_words.pck', False)
    langauge_model.load()

    print '# Loading data'
    print '  * Popular words loaded.\n  *\ttotal count :', len(popular_words)
    print '  * Distinct words loaded.'
    print '  *\tmostly not in spam:', len(distinct_words[0])
    print '  *\tmostly in spam: ', len(distinct_words[1])
    print '  * Langauge model loaded.\n  *\ttotal n_grams :', len(langauge_model.struct_3)
    
    popular_words = decode(popular_words)
    distinct_words = [decode(distinct_words[0]), decode(distinct_words[1])]

def extract_tag_features(tags, soup):
    reduce_tag_array = lambda x, y: x + len(y.string.split()) if y.string is not None else x
    features = []
    
    for tag_array in tags:
        tag_words_cnt, tag_cnt = 0, 0
        for tag in tag_array:
            tag_struct = soup.find_all(tag)
                        
            if tag_struct is not None:
                tag_cnt += len(tag_struct)
                tag_words_cnt += reduce(reduce_tag_array, tag_struct, 0)
        
        if tag_array[0] == 'title':
            features += [tag_words_cnt]
        elif tag_array[0] == 'meta' or tag_array[0] == 'img':
            features += [tag_cnt]
        else:
            features += [tag_cnt, tag_words_cnt]

    return features

# Detection spam cotent throught content analysis section 4.7/ 4.8
def pop_words_fraction(text):
    global popular_words
    cnt = []
    for N in [100, 200, 300]:
        cnt.append(reduce(lambda x, y: x + text.count(y), popular_words[:N], 0) / float(len(text)))
        cnt.append(len(set(popular_words[:N]) & set(text)) / float(N))
    return cnt

def dist_words_fraction(text):
    global distinct_words
    cnt = []
    
    for N in [200, 300]:
        for word_set in distinct_words:
            cnt.append(reduce(lambda x, y: x + text.count(y), word_set[:N], 0) / float(len(text)))
            cnt.append(len(set(word_set[:N]) & set(text)) / float(N))
    
    return cnt

def get_url_features(url):
    url_features = []
    url_features.append(url.count('.'))
    url_features.append(url.count('-') + url.count('_'))
    url_features.append(len(re.findall(r'\d', url)))
    
    return url_features

def get_tag_features(soup):
    tags = [['a'], ['img'], ['b', 'strong'], ['i', 'em'], ['meta'], ['title']]
    tag_features = extract_tag_features(tags, soup)
    
    meta = soup.find_all('meta', attrs={'name': 'keywords'})
    if meta is not None:
        cnt = 0
        for tag in meta:
            if tag.has_attr('content'):
                cnt += len(tag['content'].split())
        tag_features.append(cnt)
    return tag_features

def get_text_features(text, doc_len):
    text_features = []
    
    # text length
    text_features.append(len(text))
    if text_features[0] == 0:
        return [0] * 22
    # average_word_len
    text_features.append(reduce(lambda x, y: x + len(y), text, 0) / text_features[0])
    # compressability
    text_features.append(len(zlib.compress(''.join(text).encode('utf-8'), 1))) 
    # distinct words fraction
    text_features += dist_words_fraction(text)
    # popular words fraction
    text_features += pop_words_fraction(text)
    # visible content fraction
    text_features.append(len(''.join(text)) / float(doc_len))

    return text_features 

def html_to_features(url, raw_html, tokenizer=pymorphy_tokenizer):
    global langauge_model

    doc_len = len(raw_html)
    soup = BeautifulSoup(raw_html, "html.parser")
    
    [s.extract() for s in soup(['script', 'style'])]
    text = list(tokenizer(soup.get_text(), STOP_WORDS))
    
    features = []
    features = get_url_features(url)
    features += get_tag_features(soup)
    features += get_text_features(text, doc_len)
    # features.append(langauge_model.predict(text))

    return features
