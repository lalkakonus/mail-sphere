# coding: utf-8

acceptable = 'йцукенгшщзхъфывапролджэячсмитьбюёЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮЁ'
acceptable += 'ZXCVBNMASDFGHJKLQWERTYUIOpzxcvbnmasdfghjklqwertyuiop'

def check_word(struc, threshold):
    for symbol in struc[0]:
        if symbol not in acceptable:
            return False
    if struc[1][0] < threshold or struc[1][1] < threshold:
        return False
    return True

# words most popular words
def pop_words_sort(x):
    if type(x) is int:
        return x[1]
    return sum(x[1])
        
class distinct_sort():
    def __init__(self, doc_cnt):
        self.ratio = doc_cnt[0] / float(doc_cnt[1])
    
    # words which mostly appear in NOT spam pages
    def direct(self, x):
        return x[1][0] / (x[1][1] * self.ratio)

    # words which mostly appear in spam pages
    def reverse(self, x):
        return x[1][1] / x[1][0] * self.ratio

def get_top_words(idf, sort_function, N, threshold):
    word_list = []
    for x in idf.items():
        if check_word(x, threshold):
            word_list.append(x)
    
    word_list.sort(key=sort_function, reverse=True)
    feature_words = word_list[:N]
    
    # for j, it in enumerate(feature_words, 1):
    #     print j, it[0].decode('utf-8'), it[1]

    return [x[0] for x in feature_words]

def get_pop_words(idf, doc_cnt, N=300):
    # print 'Popular words:'
    if type(doc_cnt) is int:
        cnt = doc_cnt
    else:
        cnt = sum(doc_cnt)
    return get_top_words(idf, pop_words_sort, N, 0.03 * cnt)
    

def get_distinct_words(idf, doc_cnt, N=300, M=300):
    sort_function = distinct_sort(doc_cnt)
    if type(doc_cnt) is int:
        cnt = doc_cnt
    else:
        cnt = sum(doc_cnt)
    # print 'Distinct words:'
    return [get_top_words(idf, sort_function.direct, N, 0.03 * cnt),
            get_top_words(idf, sort_function.reverse, M, 0.03 * cnt)] 
