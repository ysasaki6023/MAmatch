# -*- coding: utf-8 -*-
import lxml.html
import requests
import csv

ofile = open("res.csv","w")
writer = csv.writer(ofile,lineterminator='\n')
writer.writerow(["date","content","url","title","type"])

allItem = []
for page in range(1,476+1):
    print page
    target_url = 'https://www.nihon-ma.co.jp/news/allnews/page/%d/'%page
    target_html = requests.get(target_url).text
    root = lxml.html.fromstring(target_html)
    for i,k in enumerate(root.cssselect('article')[1:]):
        temp = []
        for kk in k.cssselect('p'):
            if unicode(kk.text_content()).startswith(u" 続き"): continue
            if unicode(kk.text_content()).startswith(u"キーワード"): continue
            temp.append(unicode(kk.text_content()).replace(u"続きを読む→",""))
        allItem.append(temp[:2])

        kk = k.cssselect('h2.headtitle01 > a')[0]
        allItem[i].append(kk.get("href"))
        allItem[i].append(unicode(kk.text_content()))

        myList = []
        for kk in k.cssselect('div.entry-tags > p > a'):
            myList.append(unicode(kk.text_content()))
        if myList==[]: allItem[i].append(u"")
        else:          allItem[i].append(" ".join(myList))
    

    for i in allItem:
        writer.writerow([x.encode("shift-jis","ignore") for x in i])
        print i[0],i[1],i[2],i[3],i[4]
        ofile.flush()
        allItem = []
ofile.close()
