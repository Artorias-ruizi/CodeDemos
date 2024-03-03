import bibtexparser
import bibtexparser as bp
import numpy as np
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode
import pandas as pd

if __name__ == '__main__':
    with open('cite.txt') as bibfile:
        parser = BibTexParser()  # 声明解析器类
        parser.customization = convert_to_unicode  # 将BibTeX编码强制转换为UTF编码
        bibdata = bp.load(bibfile, parser=parser)  # 通过bp.load()加载

    # 输出作者和DOI
    # print(bibdata.entries[1]['title'])
    # print(bibdata.entries[1]['journal'])
    # data = pd.DataFrame(['title','author','journal'])
    # #
    # # data.append([['title'], ['author'], ['journal']])
    # data[0, 0] = 'title'
    # data[0, 1] = 'author'
    # data[0, 2] = 'journal'
    # data[1, 0] = bibdata.entries[1]['title']
    # for i in range(600):
    #     # data.append([bibdata.entries[i]['title'], bibdata.entries[i]['author'], bibdata.entries[i]['journal']])
    #     data[i + 1, 0] = bibdata.entries[i]['title']
    #     data[i + 1, 1] = bibdata.entries[i]['author']
    #     data[i + 1, 2] = bibdata.entries[i]['journal']
    #
    # print(data)
    #===========
    for i in range(len(bibdata.entries)):
        if 'journal' in bibdata.entries[i].keys():
            continue
        else:
            del bibdata.entries[i]



    num_items = len(bibdata.entries)
    data = {
        'title': [bibdata.entries[i]['title'] for i in range(num_items)],
        'journal':[bibdata.entries[i]['journal'] for i in range(num_items)],
        'author': [bibdata.entries[i]['author'] for i in range(num_items)]
    }
    df = pd.DataFrame(data)

    df.to_csv("./papers1.csv")