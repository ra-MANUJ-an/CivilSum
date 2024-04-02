import re
import pandas as pd
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)

def fix_sent_tokenization(sents, min_words, debug=False):
    '''
    Implement heuristics to improve quality of sentence tokenization.
    It will merge sentences that are too short or contain domain-specific
    paragraph markers.
    '''
    jj = 0

    def is_mergeable(sent, previous_sent):
        prev_is_short_sentence = len(previous_sent.split()) < min_words
        is_short_sentence = len(sent.split()) < min_words
        start_with_non_alpha = re.match(r'^([^\w]|\d)+.*', sent.strip())
        is_paragraph_number = re.match(r'^\d+\.$', sent.strip()) is not None
        starts_with_paragraph_number = re.match(r'^\d+\.\s.*', sent.strip()) is not None

        mergeable = sent[0].islower() or prev_is_short_sentence or is_short_sentence or start_with_non_alpha
        mergeable = mergeable and not (starts_with_paragraph_number or is_paragraph_number)
        return mergeable

    mergeable_sents = [is_mergeable(sents[ii], sents[ii-1]) for ii in range(1, len(sents))]

    while len(sents) > 1 and any(mergeable_sents):

        new_sents = [sents[0]]
        mergeable_sents = [False]

        for ii, sent in enumerate(sents[1:]):
            previous_sent = new_sents[-1].strip()
            mergeable = is_mergeable(sent, previous_sent)
            mergeable_sents.append(mergeable)
            
            if mergeable:
                new_sents[-1] = ' '.join([previous_sent, sent])
            else:
                new_sents.append(sent)

        sents = new_sents
        
        if debug:
            import rich
            
            for ii in range(len(mergeable_sents)):
                if mergeable_sents[ii]:
                    rich.print(ii-1, sents[ii-1])
                    rich.print(ii, sents[ii])

            jj += 1
            if jj > 10:
                break
    return sents


def clean_sentence(sent):
    start_with_non_alpha_pattern = re.compile("^(_|[^\w\@])*\s")
    sent = start_with_non_alpha_pattern.sub("", sent)
    sent = sent.replace("\n", " ")
    sent = sent.strip()
    return sent


def legal_sent_tokenize(text, min_words=5):
    if type(text) == str:
        sents = nltk.sent_tokenize(text)
    else:
        sents = [s for x in text for s in nltk.sent_tokenize(x)]
    sents = fix_sent_tokenization(sents, min_words=min_words)
    sents = [clean_sentence(x) for x in sents if x != "\n"]
    return sents

def find_paragraph_refs(texts):
    digit = '(?:\d\.?)'
    connectors = '(?:and|&|\,|(?:\,\s)?to)'
    start_par = '\['
    end_par = '(?:\]|\s)'
    par_pattern = f'{start_par}(Paras?\s{digit}+(?:\s?{connectors}\s?{digit}+)*){end_par}'
    par_pattern = re.compile(par_pattern)

    def get_digits(texts):
        digits = []
        for text in texts:
            text_digits = re.findall(r'\d+', text)
            digits.extend(text_digits)
        return [int(d) for d in digits]

    paragraphs = texts.apply(lambda x: re.findall(par_pattern, x) if re.search(par_pattern, x) else None)
    paragraphs = paragraphs.apply(lambda x: get_digits(x) if x else None)
    has_paragraph = len(paragraphs[~paragraphs.isnull()])
    print(f'{has_paragraph} out of {len(texts)} samples have at least one paragraph')
    return paragraphs

def find_paragraphs(text):
    par = re.split(r'[\.\s+](\d{1,2})\.\s+', text)
    return par

def process_data(data_split, data_split_paragraphs):
    par_list = []
    for itr in range(len(data_split)):
        par = find_paragraphs(data_split['text'][itr])
        tup_list = []
        if len(par) % 2 != 0:
            tup_list.append((1, par[0],0))
            for i in range(1,int(len(par)/2)+1):
                tup_list.append((int(par[2*i-1].strip()), par[2*i].strip(),0))
        else:
            for i in range(len(par)/2):
                tup_list.append((int(par[2*i].strip()), par[2*i+1].strip(),0))
        par_list.append(tup_list)

    for i in range(len(data_split_paragraphs)):
        if data_split_paragraphs[i] != None:
            for j in range(len(data_split_paragraphs[i])):
                for k in range(len(par_list[i])):
                    n, t, f = par_list[i][k]
                    if n == data_split_paragraphs[i][j]:
                        f = 1
                    par_list[i][k] = (n, t, f)

    data_split_par_list = par_list
    data_split_ranks = data_split['rank'].tolist()
    data_split_summary = data_split['summary'].tolist()

    data_split_processed = pd.DataFrame(list(zip(data_split_ranks, data_split_summary, data_split_par_list)),
                columns=['rank', 'summary', 'paragraph_label'])

    doc_id = []
    doc_summary = []
    par_number = []
    par_text = []
    label = []
    for itr in range(len(data_split_processed)):
        rank = data_split_processed['rank'][itr]
        summary = data_split_processed['summary'][itr]
        for i in range(len(data_split_processed['paragraph_label'][itr])):
            a,b,c = data_split_processed['paragraph_label'][itr][i]
            pos = b.find('.')
            if b[:pos].strip().isdigit() == True and len(b[:pos].strip()) < 3:
                a = int(b[:pos].strip())
                b = b[pos+1:]
            if a == 1 and i > 0:
                last_text = par_text[-1] + b
                par_text = par_text[:-1]
                par_text.append(last_text)
                continue
            if a != 1:
                b = str(a) + '. ' + b
            doc_id.append(rank)
            doc_summary.append(summary)
            par_number.append(a)
            par_text.append(b)
            label.append(c)

    data_split_processed_split = pd.DataFrame(list(zip(doc_id, doc_summary, par_number, par_text, label)),
                columns=['doc_id', 'doc_summary', 'par_number', 'par_text', 'label'])

    for i in range(len(data_split_paragraphs)):
        etr = data_split_ranks[i]
        if data_split_paragraphs[i] != None:
            tpl = data_split_paragraphs[i]
            for j in range(len(tpl)):
                etpl = tpl[j]
                data_split_processed_split.loc[(data_split_processed_split['doc_id'] == etr) & (data_split_processed_split['par_number'] == etpl), 'label'] = 1

    catched_list_len = data_split_processed_split['label'].tolist().count(1)
    sum_list_len = 0
    for i in range(len(data_split_paragraphs)):
        if data_split_paragraphs[i] != None:
            sum_list_len += len(data_split_paragraphs[i])

    j = 0
    joined_paragraphs_list = []
    for i in range(len(data_split_ranks)):
        etr = data_split_ranks[i]
        ets = data_split_summary[i]
        a_text = []
        while data_split_processed_split['doc_id'][j] == etr and j < len(par_number):
            a_text.append((data_split_processed_split['par_number'][j], data_split_processed_split['par_text'][j], data_split_processed_split['label'][j]))
            j += 1
            if j == len(par_number):
                break
        joined_paragraphs_list.append(a_text)

    data_split_processed_joined = pd.DataFrame(list(zip(data_split_ranks, data_split_summary, joined_paragraphs_list)),
                columns=['rank', 'summary', 'text_list'])

    oracle_paragraphs_list = []
    for i in range(len(joined_paragraphs_list)):
        paraa_text = ""
        for j in range(len(joined_paragraphs_list[i])):
            a,b,c = joined_paragraphs_list[i][j]
            if c == 1:
                paraa_text += b + " "
        oracle_paragraphs_list.append(paraa_text)

    null_string_count = 0
    for i in range(len(oracle_paragraphs_list)):
        if oracle_paragraphs_list == "":
            null_string_count += 1
    print(null_string_count)

    data_split_processed_joined_oracle = pd.DataFrame(list(zip(data_split_ranks, data_split_summary, oracle_paragraphs_list)),
                columns=['rank', 'summary', 'oracle_paragraphs'])

    return data_split_processed_joined_oracle
