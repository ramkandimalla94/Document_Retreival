#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 23:01:59 2017

@author: ram
"""
from gensim.models import doc2vec
from math import sqrt
from nltk.corpus import stopwords
from collections import namedtuple
import re
#from termcolor import colored
#import json

doc1=[]
for j in range (50):
    with open('/Users/ram/Desktop/Q/InBev/Documents/bbc/All_Files/file'+str(j+1)+'.txt', 'r') as myfile:
        data1=myfile.read().lower()
        #.replace('\n', ' ').replace(",","").replace(".","").replace("-","").replace("(","").replace(")","").replace("$","").replace('"',"").replace("'","").replace("%","")                                
        data1=re.sub('[^a-zA-Z0-9 \.]', '', data1)
        doc1.append(data1)      
docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(doc1):
    words = text.split()
    tags = [i]
    docs.append(analyzedDocument(words, tags))
    
alpha_val = 0.025        
min_alpha_val = 1e-4
passes = 15              
alpha_delta = (alpha_val - min_alpha_val) / (passes - 1)
model = doc2vec.Doc2Vec( size = 100
    , window = 300
    , min_count = 1
    , workers = 4)
model.build_vocab(docs) 
for epoch in range(passes):
    model.alpha, model.min_alpha = alpha_val, alpha_val
    model.train(docs,total_examples=model.corpus_count)
    print('Completed pass %i at alpha %f' % (epoch + 1, alpha_val))
    alpha_val -= alpha_delta

#model = doc2vec.Doc2Vec.load("/Users/ram/Desktop/Q/InBev/Codes/model50docs.model")

query = raw_input("Enter your Question: ")
filtered_query = [re.sub('[^a-zA-Z0-9 \.]', '', word.lower()) for word in query.split() if word not in stopwords.words('english')]
#.replace('\n', ' ').replace(",","").replace(".","").replace("-","").replace("(","").replace(")","").replace("$","").replace('"',"").replace("'","").replace("%","").replace("?","")  for word in query.split() if word not in stopwords.words('english')]
vec=model.infer_vector(query.lower().split())

def square_rooted(x):
    return round(sqrt(sum([a*a for a in x])),3)
 
def cosine_similarity(x,y):
   numerator = sum(a*b for a,b in zip(x,y))
   denominator = square_rooted(x)*square_rooted(y)
   return round(numerator/float(denominator),3)

cosinelist=[]

for i in range(50):
    cosinelist.append(cosine_similarity(model.docvecs[i],vec))
    
l1=sorted(cosinelist,reverse=True)
print
print('Most matched Document: file'+str(cosinelist.index(l1[0])+1)+'.txt')
print 
 

fname='file'+str(cosinelist.index(l1[0])+1)+'.txt'
with open('/Users/ram/Desktop/Q/InBev/Documents/bbc/All_Files/file'+str(cosinelist.index(l1[0])+1)+'.txt', 'r') as myfile:
    data=myfile.read().lower()
print data
     
#def highlight_many(data, filtered_query):
#    replacement= lambda match: colored(match.group(),'magenta')
#    data1 = re.sub("|".join(map(re.escape, filtered_query)), replacement, data, flags=re.I)
#    print
#    print data1

#data = 'Most matched Document: file'+str(cosinelist.index(l1[0])+1)+'.txt' + '\n'+ '\n'+ data


#highlight_many(data, filtered_query)

#with open('/Users/ram/Desktop/Q/InBev/Documents/bbc/All_Files/file'+str(cosinelist.index(l1[0])+1)+'.txt', 'r') as f:
#    content = f.read()
#    print content# Read the whole file
#    lines = content.split('.') # a list of all sentences
#    i=0 
#    for line in lines:
#        print i
#        i=i+1
#        print line

#        for z in range(len(filtered_query)):# for each sentence
#        if all(word in line for word in filtered_query):
#            
#            highlight_many(data, filtered_query)
#        else:
#            print  data

#dic={'filename':fname , 'content': data}
#json1=json.dumps(dic)
#js=json.loads(json1)


