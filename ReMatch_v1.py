# coding: utf-8
import numpy as np
import pandas as pd
import re

d = pd.read_csv("res.csv",encoding="shift-jis")
c = d["content"]

rs = []
rs.append(r"^(?P<c1>.*?)\([0-9]{4}\)は、(?P<c2>.*?)の株式を")
rs.append(r"^(?P<c1>.*?)\([0-9]{4}\)は、(?P<c2>.*?)の全株式を")
"""
rs.append(r"^.*?(?P<c1>.*?)と(?P<c2>.*?)は、")
rs.append(r"^.*?(?P<c1>.*?)は、(?P<c2>.*?)(が|と|を)")
rs.append(r"^.*?である(?P<c1>.*?)は、(?P<c2>.*?)(が|と|を)")
#rs.append(r"^.*(?P<c1>.*?)は、.*?である(?P<c2>.*?)(の|が|と)")
rs.append(r"^.*(?P<c1>.*?)、(?P<c2>.*?)(が|と|を)")

rs.append(r"^(?P<c1>.*?)は、(?P<c2>.*?)が行う")
rs.append(r"傘下の株式会社(?P<c1>.*?)は、(?P<c2>.*?)との")
rs.append(r"^(?P<c1>.*?)は、(?P<c2>.*?)より")
rs.append(r"^(?P<c1>.*?)は、(?P<c2>.*?)の株式を")
rs.append(r"^(?P<c1>.*?)は、(?P<c2>.*?)の.*?株式を")
rs.append(r"^(?P<c1>.*?)は、(?P<c2>.*?)との間で")
rs.append(r"^(?P<c1>.*?)は、(?P<c2>.*?)の.*?株式を")
#rs.append(r"^.*の海外本社である(?P<c1>.*?)は、.*?の(?P<c2>.*?)の.*?株式を")
rs.append(r"^(?P<c1>.*?)は、(?P<c2>.*?)の.*?株式")
rs.append(r"^(?P<c1>.*?)と(?P<c2>.*?)は")
rs.append(r"^(?P<c1>.*?)は、(?P<c2>.*?)の事業")
rs.append(r"^(?P<c1>.*?)は、(?P<c2>.*?)と")
#rs.append(r"^(?P<c1>.*?)は、")
#rs.append(u"傘下の株式会社")
"""
cnt = 0
bad = []
allc1 = []
allc2 = []
def cleanup(word):
    """
    word = re.sub(r"株式会社","",word)
    word = re.sub(r"の.*?株式","",word)
    word = re.sub(r"連結子会社である","",word)
    word = re.sub(r"連結子会社","",word)
    word = re.sub(r"^.*?である","",word)
    word = re.sub(r"100\%子会社である","",word)
    word = re.sub(r"100\%子会社","",word)
    """
    word = re.sub(r"\([0-9]{4}\)","",word)
    word = re.sub(r"<[0-9]{4}>","",word)
    word = re.sub(r"\（.*\）","",word)
    #word = re.sub(r"\(.*\)","",word)
    return word

def IsDrop(word):
    keyWords  = []
    keyWords += [r"[0-9]{2}",r"[0-9]社"]
    keyWords += [r"ならびに",r"並びに",r"および",r"及び",r"、",r"から",r"より"]
    keyWords += [r"事業",r"である",r"手がける",r"手掛ける",r"行う",r"運営",r"展開",r"保有",r"営む",r"提供"]
    keyWords += [r"株式を",r"全て",r"持分法",r"持株",r"発行済",r"代表取締役"]
    keyWords += [r"のため"]
    keyWords += [r"提携",r"分割",r"増資",r"子会社",r"資本関係",r"傘下",r"合弁"]
    keyWords += [r"ベトナム",r"イタリア",r"インド",r"ロシア",r"インドネシア",r"ドイツ",r"フランス",r"シンガポール",r"オーストラリア",r"カナダ",r"ニュージーランド",r"有限公司",r"韓国",r"米国",r"スペイン",r"英国",r"タイ",r"フィリピン",r"マレーシア",r"欧州",r"フィンランド",r"THAILAND",r"オランダ",r"アフリカ",r"北米",r"トルコ"]
    for i in keyWords:
        if re.search(i,word): return True
    return False

goodCount = 0
for w in c:
    w = unicode(w).encode("utf-8")
    found = False
    for r in rs:
        res = re.search(r,w)
        if res:
            cnt += 1
            c1,c2 = cleanup(res.group("c1")),cleanup(res.group("c2"))
            if IsDrop(c1) or IsDrop(c2): continue
            allc1.append(c1)
            allc2.append(c2)
            print "good",":",c1,c2
            goodCount += 1
            found = True
            break
    if not found:
        #print "bad",w
        bad.append(w)
        allc1.append("")
        allc2.append("")

sc1 = pd.Series(allc1)
sc2 = pd.Series(allc2)

d["c1"]=sc1
d["c2"]=sc2

d.to_csv("div.csv",encoding="shift-jis")
print goodCount
