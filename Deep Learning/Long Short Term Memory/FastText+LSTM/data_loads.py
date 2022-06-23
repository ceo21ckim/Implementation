from settings import *
import pandas as pd
from tqdm import tqdm
import os, gzip, pickle

FILE_NAME = dict({
    'beauty' : 'reviews_Beauty_5.json.gz', 
    'electronices' : 'reviews_Electronics_5.json.gz', 
    'health' : 'reviews_Health_and_Personal_Care_5.json.gz', 
    'sport_outdoors' : 'reviews_Sports_and_Outdoors_5.json.gz'
})



def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


if __name__ == '__main__':
    for domain in tqdm('sport_outdoors', desc = 'saving pickle file...'):
        domain_gzip = FILE_NAME[domain]
        df = getDF(os.path.join(ZIP_DIR, domain_gzip))
        df.loc[:,'reviewText'] = df.loc[:,'reviewText'].apply(lambda x: x.lower())

        with open(os.path.join(PICKLE_DIR, f'{domain}.pickle'), 'wb') as f:
            pickle.dump(df, f)

        with open(os.path.join(TXT_DIR, f'{domain}.txt'), 'w', encoding='utf8') as f:
            f.write('\n'.join(df['reviewText']))

