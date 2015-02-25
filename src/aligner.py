#!/usr/bin/env python

"""
    python align.py [options]

    options are:

    -p DIR/PREFIX   prefix for parallel data, Defaults: DIR=../ PREFIX=../hansards when running from answer directory
    -n NUMBER       number of training examples to use, Default: n = sys.maxint
"""
import optparse
import sys
import os.path
import math
from collections import defaultdict
import numpy as np
import pickle
from numpy import zeros,sum,ones
#from collections import Counter
from math import log
import time
import multiprocessing as mp
from multiprocessing import Process, Value, Array
import ctypes as ct


optparser = optparse.OptionParser()
optparser.add_option("-p", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)
#i_data = "project/hansards.it"
#es_data = "project/hansards.es"

test_f_data = f_data

test_e_data = e_data



if not ( os.path.isfile(f_data) and os.path.isfile(e_data) ):
    print >>sys.stderr, __doc__.strip('\n\r')
    sys.exit(1)

sys.stderr.write("Training with Dice's coefficient...")
#bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
#tritext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data), open(es_data))[:opts.num_sents]]
#tritext_fite = [[sentence.strip().split() for sentence in pair] for pair in zip(open(e_data), open(f_data), open(es_data))[:opts.num_sents]]

bitext_fe = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
bitext_ef = [[sentence.strip().split() for sentence in pair] for pair in zip(open(e_data), open(f_data))[:opts.num_sents]]

#bitext_ei = [[sentence.strip().split() for sentence in pair] for pair in zip(open(e_data), open(es_data))[:opts.num_sents]]
#bitext_if = [[sentence.strip().split() for sentence in pair] for pair in zip(open(es_data), open(f_data))[:opts.num_sents]]

#bitext_ite = [[sentence.strip().split() for sentence in pair] for pair in zip(open(es_data), open(e_data))[:opts.num_sents]]
#bitext_fit = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(es_data))[:opts.num_sents]]

bitext_test = [[sentence.strip().split() for sentence in pair] for pair in zip(open(test_f_data), open(test_e_data))[:opts.num_sents]]
#print bitext[0]
#print len(bitext)
f_count = defaultdict(int)
e_count = defaultdict(int)
it_count = defaultdict(int)

fe_count = defaultdict(int)
ef_count = defaultdict(int)
itf_count = defaultdict(int)
eit_count = defaultdict(int)
ite_count = defaultdict(int)
fit_count = defaultdict(int)


for (n, (f, e)) in enumerate(bitext_fe):
  for f_i in set(f):
    f_count[f_i] += 1
    for e_j in set(e):
      fe_count[(f_i,e_j)] += 1
      ef_count[(e_j,f_i)] += 1
#    for i_j in set(it):
#        itf_count[(i_j,f_i)] += 1
#        fit_count[(f_i,i_j)] += 1
  for e_j in set(e):
    e_count[e_j] += 1
#    for i_i in set(it):
#        eit_count[(e_j,i_i)] += 1
#        ite_count[(i_i,e_j)] += 1
#  for it_j in set(it):
#      it_count[it_j] += 1
#  if n % 500 == 0:
#    sys.stderr.write(".")


#print fe_count[('je','I')]

def countWordsInASentence(Sentence):
    countMap = dict()
    for w in Sentence:
        if countMap.has_key(w):
            countMap[w] = countMap[w] + 1
        else:
            countMap[w] = 1
    return countMap
def EM_IBM1(source_count, target_count, st_count,bitext):
    c = defaultdict(int)
    total = defaultdict(int)
    t = defaultdict(int)
    
    #initialize t uniformly
    for (source,target) in st_count:
        t[(source,target)] = 1.0/len(source_count)
    for cnt in range(10):
        c = defaultdict(int)
        total = defaultdict(int)
        
        for (S,D) in bitext:
            for s_i in S:
                Z = 0 #normalizer
                for d_j in D:
                    Z += t[(s_i,d_j)]
                for d_j in D:
                    c[(s_i,d_j)] += t[(s_i,d_j)]/Z
                    total[d_j] += t[(s_i,d_j)]/Z
        for (s,d) in st_count:
            t[(s,d)] = c[(s,d)]/total[d]
    return t
    
#Maybe we should remove this part for performance IMPORTANT
def EM_source_target(source_count, target_count, st_count,bitext):
    c = defaultdict(int)
    total = defaultdict(int)
    t = defaultdict(int)
    #Total_denominator = defaultdict(int)
    
    #for f in f_count:
    #    for e in e_count:
    #        t[(e,f)] = 1.0/len(e_count) #This is t(e|f) initialized uniformly
    for (source,target) in st_count:
        t[(target,source)] = 1.0/len(target_count)
    #t_e_f =  1.0/len(e_count)
    #print 'by', len(e_count), ' ', len(f_count)
    for cnt in range(10):
    #    for f in f_count:
    #        for e in e_count:
    #            c[(e,f)] = 0.0
    #        total[f] = 0.0
    #        
        c = defaultdict(int)
        total = defaultdict(int)
        
        line = 0
        for (S,D) in bitext:
            line = line + 1
            uniqTargetWords = countWordsInASentence(D)
            uniqSourceWords = countWordsInASentence(S)
            for (j,d) in enumerate(uniqTargetWords):
                n_d = uniqTargetWords.get(d)
                Total = 0
                
                for s in uniqSourceWords:
                    Total += t[(d,s)]*n_d
                for (i,s) in enumerate(uniqSourceWords):
                    n_s = uniqSourceWords.get(s)
                    if (Total == 0):
                        print line
                    rhs = (t[(d,s)]*n_d*n_s)/Total
                    c[(d,s)] += rhs
                    total[s] += rhs
        for (s,d) in st_count:
            t[(d,s)] = c[(d,s)]/total[s]
    
#    for (S,D) in bitext:
#        uniqTargetWords = Counter(D)
#        uniqSourceWords = Counter(S)
#        for (i,d) in enumerate(uniqTargetWords):
#            n_d = uniqTargetWords.get(d)
#            Total_denominator = 0
#            
#            for s in uniqSourceWords:
#                Total_denominator += t[(d,s)]*n_d
#            for (j,s) in enumerate(uniqSourceWords):
#                n_s = uniqSourceWords.get(s)
#                p[(i,j)] = t[(d,s)]*1.0/Total_denominator
        
       
    return t
#This EM is an implementation including delta notations
#Please refer to Michael Collins notes 
#delta is posterior probability
def EM_source_target_MC(source_count, target_count, st_count,bitext):
    c = defaultdict(int)
    total = defaultdict(int)
    t = defaultdict(int)
    delta = defaultdict(int)
    for (source,target) in st_count:
        t[(target,source)] = 1.0/len(target_count)
    for cnt in range(11):
        c = defaultdict(int)
        total = defaultdict(int)
        
        line = 0
        for (S,D) in bitext:
            line = line + 1
            uniqTargetWords = countWordsInASentence(D)
            uniqSourceWords = countWordsInASentence(S)
            for (j,d) in enumerate(uniqTargetWords):
                n_d = uniqTargetWords.get(d)
                Total = 0
                
                for s in uniqSourceWords:
                    Total += t[(d,s)]*n_d
                for (i,s) in enumerate(uniqSourceWords):
                    n_s = uniqSourceWords.get(s)
                    if (Total == 0):
                        print line
                    rhs = (t[(d,s)]*n_d*n_s)/Total
                    c[(d,s)] += rhs
                    total[s] += rhs
                    delta[(line,i,j)] = rhs
        for (s,d) in st_count:
            t[(d,s)] = c[(d,s)]/total[s]
       
    return delta

def compute_posterior(S,D,t,j):
    total = 0.0
    
    for (i,s) in enumerate(S):
        total += t[(D[j],s)]
        #print D[j],'',s
    return total
def compute_ibm_posterior(S,D,t,j):
    total = 0.0
    
    for (i,d) in enumerate(D):
        total += t[(d,S[j])]
        #print D[j],'',s
    return total
#IBM Model 2 started ...
def IBM2_ST(source_count, target_count, st_count,bitext):
    #c = defaultdict(int)
    #total = defaultdict(int)
    t = defaultdict(int)        #THIS t should have been initialized to IBM1 parameters
    for s in range(10):
    #    for f in f_count:
    #        for e in e_count:
    #            c[(e,f)] = 0.0
    #        total[f] = 0.0
    #        
        count = defaultdict(int)
        total = defaultdict(int)
        count_a = defaultdict(int)        
        total_a = defaultdict(int)        
        s_total = defaultdict(int)
        a = defaultdict(int)
        
        for (source,target) in st_count:
            t[(target,source)] = 1.0/len(target_count)
        
        for (F,E) in bitext:
            l_e = len(E)
            l_f = len(F)
            #compute normalization
            for (j,e) in enumerate(E):
                s_total[e] = 0
                for (i,f) in enumerate(F):
                    if (not a.has_key((i,j,l_e,l_f))):
                        a[(i,j,l_e,l_f)] = 1./(l_f+1)
                    s_total[e] += t[(e,f)]*a[(i,j,l_e,l_f)]
            
            for (j,e) in enumerate(E):
                for (i,f) in enumerate(F):
                    if (not a.has_key((i,j,l_e,l_f))):
                        a[(i,j,l_e,l_f)] = 1./(l_f+1)
                    c = t[(e,f)]*a[(i,j,l_e,l_f)]/s_total[e]
                    count[(e,f)] += c
                    total[f] += c
                    count_a[(i,j,l_e,l_f)] += c
                    total_a[(j,l_e,l_f)] += c
                    
        #estimate probabilities
        #t = defaultdict(int)
        a = defaultdict(int)
            
        for (e,f) in t:
            t[(e,f)] = 1.0*count[(e,f)]/total[f]
        for (i,j,l_e,l_f) in count_a:
            a[(i,j,l_e,l_f)] = 1.0*count_a[(i,j,l_e,l_f)]/total_a[(j,l_e,l_f)]
    return (t,count_a)

def bridge(p_sg, p_gd, tritext):    #sg is source to bridge and gd is bridge to dest
    
    p_sd = defaultdict(int)
    
    for (n, (D, S, G)) in enumerate(tritext): # order in tritext 
        for (j, s_j) in enumerate(S):
            for (i, d_i) in enumerate(D):
                total = 0
                for (k, g_k) in enumerate(G):
                    total += p_sg[(g_k,s_j)]*p_gd[(d_i,g_k)]
                p_sd[(d_i,s_j)] = total
    return p_sd

def bridge_posterior(p_sg, p_gd, tritext):    #sg is source to bridge and gd is bridge to dest
    
    p_sd = defaultdict(int)
    
    for (n, (D, S, G)) in enumerate(tritext): # order in tritext 
        for (j, s_j) in enumerate(S):
            for (i, d_i) in enumerate(D):
                total = 0
                for (k, g_k) in enumerate(G):
                    total += p_sg[(g_k,s_j,n)]*p_gd[(d_i,g_k,n)]
                p_sd[(d_i,s_j,n)] = total
    return p_sd
    

#(t2,p2) = EM_source_target(f_count,e_count,fe_count,bitext_fe)
#(t,P) = IBM2_ST(e_count,f_count,ef_count,bitext_ef)

def fill_in_posterior(bitext,t):
    p_sd = defaultdict(int)
    for (n,(S, D)) in enumerate(bitext):
        for (j,d) in enumerate(D):
            denom = compute_posterior(S,D,t,j)
            for (i,s_i) in enumerate(S): 
                if denom != 0:
                    p_sd[(d,s_i)] = t[(d,s_i)]*1.0/denom
                else:
                    p_sd[(d,s_i)] = 0
    return p_sd
def fill_in_posterior_ibm(bitext,t):
    p_sd = defaultdict(int)
    for (n,(S, D)) in enumerate(bitext):
        for (j,s) in enumerate(S):
            denom = compute_ibm_posterior(S,D,t,j)
            for (i,d) in enumerate(D): 
                if denom != 0:
                    p_sd[(d,s)] = t[(d,s)]*1.0/denom
                else:
                    p_sd[(d,s)] = 0
    return p_sd
def fill_in_posterior2(bitext,t):
    p_sd = defaultdict(int)
    for (n,(S, D)) in enumerate(bitext):
        i = 0
        for s_i in S: 
            j = 0
            for d in D:
                #we could have another loop on D above to save time
                denom = compute_posterior(S,D,t,j)
                #print denom
                if denom != 0:
                    p_sd[(d,s_i)] = t[(d,s_i)]*1.0/denom
                else:
                    p_sd[(d,s_i)] = 0
                j += 1
            i += 1
    return p_sd
def fill_in_posteriorMC(bitext,t):
    delta = defaultdict(int)
    for (n,(S, D)) in enumerate(bitext,start=1):
        for (j,d) in enumerate(D):
            denom = compute_posterior(S,D,t,j)
            for (i,s_i) in enumerate(S):
                if denom != 0:
                    delta[(n,i,j)] = t[(d,s_i)]*1.0/denom
                else:
                    delta[(n,i,j)] = 0
    return delta

def intersect_SD_with_DS_Models(bitext,p_sd,p_ds):
    for (n,(S, D)) in enumerate(bitext):
        set1 = set()
        set2 = set()
        i = 0
        for s_i in S: 
            max_p = 0
            argmax = -1
            j = 0
            for d in D:
                if p_sd[(s_i,d)] > max_p:
                    max_p = p_sd[(s_i,d)]
                    argmax = j  
                j += 1
            set1.add((i,argmax))
            i += 1
        i = 0
        for d_i in D: 
            max_p = 0
            argmax = -1
            j = 0
            for s in S:
                if p_ds[(d_i,s)] > max_p:
                    max_p = p_ds[(d_i,s)]
                    argmax = j  
                j += 1
            set2.add((argmax,i)) 
            i += 1
            intersect = set1.intersection(set2)
        for (i_s,argmax_s) in intersect:
            sys.stdout.write("%i-%i " % (i_s,argmax_s))
        sys.stdout.write("\n")
        
def new_measure_with_posterior(bitext,p_sd,p_ds):
    for (n,(S, D)) in enumerate(bitext):
        set1 = set()
        set2 = set()
        i = 0
        for s_i in S: 
            max = 0
            argmax = -1
            j = 0
            for d in D:
                #we could have another loop on D above to save time
                #denom = compute_posterior(S,D,t,j)
                #print denom
#                if denom != 0:
#                    p_ij = t[(d,s_i)]*1.0/denom
#                else:
#                    p_ij = 0
                if p_sd[(d,s_i)] > max:
                    max = p_sd[(d,s_i)]
                    argmax = j  
                j += 1
            set1.add((i,argmax))#UPDATE JULY 30 +1 is deleted from both
#           sys.stdout.write("%i-%i " % (i+1,argmax+1))
            i += 1
#       sys.stdout.write("\n")
        i = 0
        for d_i in D: 
            max = 0
            argmax = -1
            j = 0
            for s in S:
                if p_ds[(s,d_i)] > max:
                    max = p_ds[(s,d_i)]
                    argmax = j  
                j += 1
            set2.add((argmax,i)) #UPDATE JULY 30 +1 is deleted from both
    #        sys.stdout.write("%i-%i " % (i,argmax))
            i += 1
            intersect = set1.intersection(set2)
        for (i_s,argmax_s) in intersect:
            sys.stdout.write("%i-%i " % (i_s,argmax_s))
        sys.stdout.write("\n")
        
def new_measure_with_posterior_MC(bitext,delta_sd,delta_ds):
    for (n,(S, D)) in enumerate(bitext,start=1):
        set1 = set()
        set2 = set()
        i = 0
        for s_i in S: 
            max = 0
            argmax = 0
            j = 0
            for d in D:
                #we could have another loop on D above to save time
                #denom = compute_posterior(S,D,t,j)
                #print denom
#                if denom != 0:
#                    p_ij = t[(d,s_i)]*1.0/denom
#                else:
#                    p_ij = 0
                if delta_sd[(n,i,j)] > max:
                    max = delta_sd[(n,i,j)]
                    argmax = j  
                j += 1
            set1.add((i,argmax))#UPDATE JULY 30 +1 is deleted from both
 #           sys.stdout.write("%i-%i " % (i+1,argmax+1))
            i += 1
 #       sys.stdout.write("\n")
        i = 0
        for d_i in D: 
            max = 0
            argmax = 0
            j = 0
            for s in S:
                if delta_ds[(n,i,j)] > max:
                    max = delta_ds[(n,i,j)]
                    argmax = j  
                j += 1
            set2.add((argmax,i)) #UPDATE JULY 30 +1 is deleted from both
    #        sys.stdout.write("%i-%i " % (i,argmax))
            i += 1
            intersect = set1.intersection(set2)
    #print n,'\t',intersect
        for (i_s,argmax_s) in intersect:
            sys.stdout.write("%i-%i " % (i_s,argmax_s))
        sys.stdout.write("\n")
        
def measure_with_posterior(bitext,t):
    for (n,(S, D)) in enumerate(bitext):
        set1 = set()
        set2 = set()
        i = 0
        for s_i in S: 
            max = 0
            argmax = -1
            j = 0
            for d in D:
                #we could have another loop on D above to save time
                denom = compute_posterior(S,D,t,j)
                #print denom
                if denom != 0:
                    p_ij = t[(d,s_i)]*1.0/denom
                else:
                    p_ij = 0
                if p_ij > max:
                    max = p_ij
                    argmax = j  
                j += 1
            set1.add((i+1,argmax+1))
            sys.stdout.write("%i-%i " % (i+1,argmax+1))
            i += 1
        sys.stdout.write("\n")

def measure(bitext,t):
    for (S, D) in bitext:
        set1 = set()
        set2 = set()
        i = 0
        for s_i in S: 
            max = 0
            argmax = -1
            j = 0
            for d in D:
                if t[(d,s_i)] > max:
                    max = t[(d,s_i)]
                    argmax = j  
                j += 1
            set1.add((i+1,argmax+1))
            sys.stdout.write("%i-%i " % (i+1,argmax+1))
            i += 1
        sys.stdout.write("\n")
#        i = 0
#        for d_i in D: 
#            max = 0
#            argmax = -1
#            j = 0
#            for s in S:
#                if t2[(s,d_i)] > max:
#                    max = t2[(s,d_i)]
#                    argmax = j  
#                j += 1
#            set2.add((argmax+1,i+1))
#    #        sys.stdout.write("%i-%i " % (i,argmax))
#            i += 1
#            intersect = set1.intersection(set2)
#        for (i_s,argmax_s) in intersect:
#            sys.stdout.write("%i-%i " % (i_s+1,argmax_s+1))
#        sys.stdout.write("\n")


#measure(bitext_test,t,t)

#
def measure_interpolation(bitext,t,t2,t3):
    for (S, D) in bitext:
        set1 = set()
        set2 = set()
        i = 0
        for s_i in S:
            max = 0
            argmax = -1
            j = 0
            for d in D:
                if (t[(d,s_i)]+t2[(d,s_i)]+t3[(d,s_i)]) > max:
                    max = t[(d,s_i)]+t2[(d,s_i)]+t3[(d,s_i)]
                    argmax = j
                j += 1
            set1.add((i+1,argmax+1))
            sys.stdout.write("%i-%i " % (i+1,argmax+1))
            i += 1
        sys.stdout.write("\n")

def measure_interpolation_both_direction(bitext,t,t2,t3,t4,t5,t6):
    for (S, D) in bitext:
        set1 = set()
        set2 = set()
        i = 0
        for s_i in S:
            max = 0
            argmax = -1
            j = 0
            for d in D:
                if (t[(d,s_i)]+t2[(d,s_i)]+t3[(d,s_i)]) > max:
                    max = t[(d,s_i)]+t2[(d,s_i)]+t3[(d,s_i)]
                    argmax = j
                j += 1
            set1.add((i,argmax))
            #sys.stdout.write("%i-%i " % (i+1,argmax+1))
            i += 1
        #sys.stdout.write("\n")
        i = 0
        for d_i in D:
            max = 0
            argmax = -1
            j = 0
            for s in S:
                if (t4[(s,d_i)]+t5[(s,d_i)]+t6[(s,d_i)]) > max:
                    max = t4[(s,d_i)]+t5[(s,d_i)]+t6[(s,d_i)]
                    argmax = j
                j += 1
            set2.add((argmax,i))
    #        sys.stdout.write("%i-%i " % (i,argmax))
            i += 1
            intersect = set1.intersection(set2)
        for (i_s,argmax_s) in intersect:
            sys.stdout.write("%i-%i " % (i_s,argmax_s))
        sys.stdout.write("\n")

def writePIntoFile(p_sd):
    outFile = 'p_out.txt'
    w = open(outFile, 'w')
    for (d,s) in p_sd:
        w.write(d+' '+s+' '+str(p_sd[(d,s)])+'\n')
    w.close()

def maxTargetSentenceLength(bitext):
    maxLength = 0
    for (s,d) in bitext:
        if len(d) > maxLength:
            maxLength = len(d)
    return maxLength

def readP_sd(readFile):
    p_sd = defaultdict(int)
    f = open(readFile, 'r')
    for line in f:
        l = line.split()
        d = l[0]
        s = l[1]
        p = l[2]
        p_sd[(d,s)] = float(p)
    return p_sd


#T is the length of the observation sequence
def forward(a,b,pi,y,N,T): #N is the number of states
    alpha = dict()
    for i in range(1,N+1):
        alpha[(i,1)] = pi[i]*b[(i,y[0])]
        
    for t in range(1,T):
        for j in range(1,N+1):
            total = 0
            for i in range(1,N+1):
                total += alpha[(i,t)]*a[(i,j)]
            alpha[(j,t+1)] = b[(j,y[t])]*total
    return alpha
#T is the length of the observation sequence
def forward_with_t(a,pi,y,N,T,d, t_table): #N is the number of states and d is the target sentence
    # y is the source sentence 
    # d is the dest sentence
    alpha = dict()
    for i in range(1,N+1):
        alpha[(i,1)] = pi[i]*t_table[(y[0],d[i-1])]
        
    for t in range(1,T):
        for j in range(1,N+1):
            total = 0
            for i in range(1,N+1):
                total += alpha[(i,t)]*a[(i,j)]
            alpha[(j,t+1)] = t_table[(y[t],d[j-1])]*total
    return alpha

def forward_with_t_scaled(a,pi,y,N,T,d, t_table): #N is the number of states and d is the target sentence
    # y is the source sentence 
    # d is the dest sentence
    c_scaled = ones(T+1)
    alpha_hat = zeros((N+1,T+1))
    total_alpha_double_dot = 0
    for i in range(1,N+1):
        alpha_hat[(i,1)] = pi[i]*t_table[(y[0],d[i-1])]
        total_alpha_double_dot += alpha_hat[(i,1)]
    c_scaled[1] = 1.0/total_alpha_double_dot
    alpha_hat[:,1] = c_scaled[1]*alpha_hat[:,1]
#     for i in range(1,N+1):
#         alpha_hat[(i,1)] = c_scaled[1]*alpha_hat[(i,1)]
    for t in range(1,T):
        total_alpha_double_dot = 0
        for j in range(1,N+1):
            total = 0
            for i in range(1,N+1):
                total += alpha_hat[(i,t)]*a[(i,j)]
            alpha_hat[(j,t+1)] = t_table[(y[t],d[j-1])]*total
            total_alpha_double_dot += alpha_hat[(j,t+1)]
        c_scaled[t+1] = 1.0/total_alpha_double_dot
        #for i in range(1,N+1):
        #    alpha_hat[(i,t+1)] = c_scaled[t+1]*alpha_hat[(i,t+1)]
        alpha_hat[:,t+1] = c_scaled[t+1]*alpha_hat[:,t+1]
        
    return (alpha_hat,c_scaled)

def backward(a,b,pi,y,N,T):
    beta = dict()
    for i in range(1,N+1):
        beta[(i,T)] = 1
    for t in range(T-1,0,-1):
        for i in range(1,N+1):
            total = 0
            for j in range(1,N+1):
                total += beta[(j,t+1)]*a[(i,j)]*b[(j,y[t])]
            beta[(i,t)] = total
    return beta
def backward_with_t(a,pi,y,N,T,d, t_table):
    beta = dict()
    for i in range(1,N+1):
        beta[(i,T)] = 1
    for t in range(T-1,0,-1):
        for i in range(1,N+1):
            total = 0
            for j in range(1,N+1):
                total += beta[(j,t+1)]*a[(i,j)]*t_table[(y[t],d[j-1])]
            beta[(i,t)] = total
    return beta
def backward_with_t_scaled(a,pi,y,N,T,d, t_table,c_scaled):
    beta_hat = zeros((N+1,T+1))
#     for i in range(1,N+1):
#         beta_hat[(i,T)] = c_scaled[T]
    beta_hat[:,T] = c_scaled[T]
    for t in range(T-1,0,-1):
        for i in range(1,N+1):
            total = 0
            for j in range(1,N+1):
                total += beta_hat[(j,t+1)]*a[(i,j)]*t_table[(y[t],d[j-1])]
            beta_hat[(i,t)] = c_scaled[t]*total
    return beta_hat
def initializeUniformly(N): # K is the number of all possible values for Ys
    #a = zeros((N+1,N+1))
    #pi = zeros(N+1)
    a_array = Array(ct.c_double,((N+1)*(N+1)))
    a_array2 = np.frombuffer(a_array.get_obj()) # mp_arr and arr share the same memory
    a = a_array2.reshape((N+1,N+1)) # b and arr share the same memory

    #pi = zeros(N+1)
    pi = Array('d', N+1)

    for i in range(1,N+1):
        pi[i] = 1.0/N
    for i in range(1,N+1):
        for j in range(1,N+1):
            a[(i,j)] = 1.0/N
#     for j in range(1,N+1):
#         for y_t in yValues:
#             b[(j,y_t)] = t[(y_t,)]
    return (a,pi)

def compute_c(): #Indirect HMM-based hypothesis alignment  by He et al.
    k = 2
    c = defaultdict(int)
    for d in range(-5,8):
        c[d] = 1.0/pow(1+abs(d-1),k)
    return c

def initializeBasedOnC(N):
    a = zeros((N+1,N+1))
    pi = zeros(N+1)
     
    c = compute_c()
     
    total = defaultdict(int)
    for i in range(1,N+1):
        pi[i] = 1.0/N
    for l in range(1,N+1):
        for i in range(1,N+1):
            total[i] += c[l-i]
    for i in range(1,N+1):
        for j in range(1,N+1):
            a[(i,j)] = c[j-i]/total[i]

    return (a,pi)



#computes all possible values for Y variable
def computeYValues(Y):
    yValues = set()
    for y_i in Y:
        for y_ij in y_i:
            yValues.add(y_ij)
    return yValues

def map_bitext_to_int(sd_count):
    #index = zeros(len(sd_count)) #index vector
    index = defaultdict(int)
    biword = defaultdict(int)
    for (i,(s,d)) in enumerate(sd_count):
        index[(s,d)] = i
        biword[i] = (s,d)
    return (index,biword)

def Expectation2(lock, t_table, N, Y, sd_size, indexMap, iterations, totalGammaOverAllObservations, totalGammaDeltaOverAllObservations_t, totalGamma1OverAllObservations, totalC_j_Minus_iOverAllObservations, totalC_l_Minus_iOverAllObservations,start,end,a,pi,logLikelihood, lastLogLikelihood):
    #print 'Expectation2'
    for y, x in Y[start:end]: #y is the source sentence and x is the target sentence
        T = len(y)
        N = len(x)
        c = defaultdict(int)
    #MAYBE we have to initialize a here???????????
    #You probably need to have N = I here ********????
    #This if else is just for the sake of test to see b and t_table work similarly or not as anoop claimed
        if iterations == 0:
            a, pi = initializeBasedOnC(N)
        alpha_hat, c_scaled = forward_with_t_scaled(a, pi, y, N, T, x, t_table) #BE careful on indexing on x
        beta_hat = backward_with_t_scaled(a, pi, y, N, T, x, t_table, c_scaled) #else:
        #    alpha = forward(a, b, pi, y, N, T)
        #    beta = backward(a, b, pi, y, N, T)
        gamma = np.zeros((N + 1, T + 1))
        xi = zeros((N + 1, N + 1, T + 1))
        #gamma = alpha_hat * beta_hat / c_scaled
        #logLikelihood += np.sum(np.log(c_scaled))
        #np.res
        #tempSum = (np.sum(gamma,axis=1))
        #tempSum.resize(N_max+1)
        #totalGammaOverAllObservations += tempSum
        #            total = dict()
        for t in range(1, T):
            #print 'c_scaled ',c_scaled[t]
            with lock:
            	logLikelihood.value += -(log(c_scaled[t]))
#                 total[t] = 0
#                 for j in range(1,N+1):
#                     total[t] += alpha_hat[(j,t)]*beta_hat[(j,t)]
#                 #total[t] computed here is P(X) or P_l
            #totalOverAllObservations[t] += total[t]
            for i in range(1, N + 1):
                gamma[(i,t)] = (alpha_hat[(i,t)]*beta_hat[(i,t)])/c_scaled[t]
                #totalAlphaBetaOverAllObeservations[(i,t)] += (alpha[(i,t)]*beta[(i,t)])
                #totalGamma[i] += gamma[(i,t)]
#                    totalGammaOverAllObservationsMinusLast[i] += gamma[(i,t)]
                with lock:
                    totalGammaOverAllObservations[i] += gamma[i, t]
#                    totalGammaDeltaOverAllObservations[(i,y[t-1])] += gamma[(i,t)]
                #totalGammaDeltaOverAllObservations_t[(i,y[t-1],x[i-1])] += gamma[(i,t)]
                #with totalGammaDeltaOverAllObservations_t.get_lock():
                #Computing the address of 1-D araray
                address = (i * sd_size) + indexMap[(y[t - 1], x[i - 1])]
                with lock:
                    totalGammaDeltaOverAllObservations_t[address] += gamma[i, t]
                #totalGammaDeltaOverAllObservations_t[i, indexMap[(y[t - 1], x[i - 1])]] += gamma[i, t]
#SLOW and Correct version with manager.dict
#                 if (i, indexMap[(y[t - 1], x[i - 1])]) in totalGammaDeltaOverAllObservations_t: 
#                 	totalGammaDeltaOverAllObservations_t[i, indexMap[(y[t - 1], x[i - 1])]] = totalGammaDeltaOverAllObservations_t[i, indexMap[(y[t - 1], x[i - 1])]]+ gamma[i, t] #totalGammaDeltaOverAllObservations_t_overall_states[(y[t-1],x[i-1])] += gamma[(i,t)] #You can remove the upper variable
#         	else:
# 			totalGammaDeltaOverAllObservations_t[i, indexMap[(y[t - 1], x[i - 1])]] = gamma[i, t] 
                #for j in range(1,N+1):
                #    xi[(i,j,t)] = (alpha[(i,t)]*a[(i,j)]*b[(j,y[t+1])]*beta[(j,t+1)])/total[t]
        t = T
        #             total[t] = 0.0
        #             for j in range(1,N+1):
        #                 total[t] += alpha[(j,t)]*beta[(j,t)]
        #Here value of total[T] is computed
        with lock: 
            logLikelihood.value += -(log(c_scaled[t]))
        for i in range(1, N + 1): #
            gamma[(i,t)] = (alpha_hat[(i,t)]*beta_hat[(i,t)])/c_scaled[t]
            with lock:
                totalGammaOverAllObservations[i] += gamma[i, t] #               totalGammaDeltaOverAllObservations[(i,y[t-1])] += gamma[(i,t)]
            #totalGammaDeltaOverAllObservations_t[(i,y[t-1],x[i-1])] += gamma[(i,t)]
            address = (i * sd_size) + indexMap[(y[t - 1], x[i - 1])]
            with lock:
                totalGammaDeltaOverAllObservations_t[address] += gamma[i, t]
            #totalGammaDeltaOverAllObservations_t[i, indexMap[(y[t - 1], x[i - 1])]] += gamma[i, t]
            #with totalGammaDeltaOverAllObservations_t.get_lock():
#SLOW and Correct version with manager.dict
#             if (i, indexMap[(y[t - 1], x[i - 1])]) in totalGammaDeltaOverAllObservations_t: 
#             	totalGammaDeltaOverAllObservations_t[i, indexMap[(y[t - 1], x[i - 1])]] = totalGammaDeltaOverAllObservations_t[i, indexMap[(y[t - 1], x[i - 1])]]+ gamma[i, t] #totalGammaDeltaOverAllObservations_t_overall_states[(y[t-1],x[i-1])] += gamma[(i,t)]
#             else:
# 		totalGammaDeltaOverAllObservations_t[i, indexMap[(y[t - 1], x[i - 1])]] = gamma[i, t]
	#print 'gamma',gamma
        for t in range(1, T):
            for i in range(1, N + 1):
                for j in range(1, N + 1):
#                        if iterations == 0:
                    xi[i, j, t] = alpha_hat[(i, t)] * a[(i, j)] * t_table[(y[t], x[j - 1])] * beta_hat[(j, t + 1)] #                        else:
        
#                            xi[(i,j,t)] = (alpha[(i,t)]*a[(i,j)]*b[(j,y[t])]*beta[(j,t+1)])/total[t]
#                        totalXiOverAllObservations[(i,j)] += xi[(i,j,t)]
        for i in range(1, N + 1):
            
            totalGamma1OverAllObservations[i] += gamma[i, 1]
        
        for d in range(-N - 1, N+1):
            #c[d] = 0
            for t in range(1, T):
                for i in range(1, N + 1):
                    if i + d <= N and i + d >= 1:
                        c[d] += xi[i, i + d, t]
        
        #print 'c',c
        #Liang et al. suggestion
#             for d in c:
#                 if d < -7:
#                     c[-7] += c[d]
#                     c[d] = 0
#                 if d > 7:
#                     c[7] += c[d]
#                     c[d] = 0
	#print 'totalC_j_Minus_iOverAllObservations',totalC_j_Minus_iOverAllObservations
        #print 'N',N
	for i in range(1, N + 1):
            for j in range(1, N + 1): #                     if j-i >= 7:
#                         totalC_j_Minus_iOverAllObservations[(i,j)] += c[7]
#                     elif j-i <= -7:
#                         totalC_j_Minus_iOverAllObservations[(i,j)] += c[-7]
#                     else:
                with lock:
                    totalC_j_Minus_iOverAllObservations[(i,j)] += c[j - i]
                #address = i*(N+1)+j
                #with lock:
                #    totalC_j_Minus_iOverAllObservations[address] += c[j - i]
#SLOW but correct
#             	if (i,j) not in totalC_j_Minus_iOverAllObservations:
# 			totalC_j_Minus_iOverAllObservations[(i,j)] = c[j - i]
# 		else:
# 			totalC_j_Minus_iOverAllObservations[(i, j)] += c[j - i]
            
            for l in range(1, N + 1): #                     if l-i >= 7:
#                         totalC_l_Minus_iOverAllObservations[i] += c[7]
#                     elif l-i <= -7:
#                         totalC_l_Minus_iOverAllObservations[i] += c[-7]
#                     else:
                with lock:
                    totalC_l_Minus_iOverAllObservations[i] += c[l - i]
    
    
    #return N, i, a, pi, j
def Expectation(lock,t_table, N, Y, indexMap, iterations, totalGammaOverAllObservations, totalGammaDeltaOverAllObservations_t, totalGamma1OverAllObservations, totalC_j_Minus_iOverAllObservations, totalC_l_Minus_iOverAllObservations,start,end,a,pi,logLikelihood, lastLogLikelihood):
    #print start
    #print end
    for y, x in Y[start:end]: #y is the source sentence and x is the target sentence
        T = len(y)
        N = len(x)
        c = defaultdict(int)
    #MAYBE we have to initialize a here???????????
    #You probably need to have N = I here ********????
    #This if else is just for the sake of test to see b and t_table work similarly or not as anoop claimed
        if iterations == 0:
            a, pi = initializeUniformly(N)
        
        alpha_hat, c_scaled = forward_with_t_scaled(a, pi, y, N, T, x, t_table) #BE careful on indexing on x
        print 'c_scaled',c_scaled[1],c_scaled[3]
        beta_hat = backward_with_t_scaled(a, pi, y, N, T, x, t_table, c_scaled) #else:
        #    alpha = forward(a, b, pi, y, N, T)
        #    beta = backward(a, b, pi, y, N, T)
        gamma = np.zeros((N + 1, T + 1))
        xi = zeros((N + 1, N + 1, T + 1))
        #gamma = alpha_hat * beta_hat / c_scaled
        #logLikelihood += np.sum(np.log(c_scaled))
        #np.res
        #tempSum = (np.sum(gamma,axis=1))
        #tempSum.resize(N_max+1)
        #totalGammaOverAllObservations += tempSum
        #            total = dict()
        for t in range(1, T):
            #print 'c_scaled ',c_scaled[t]
            with lock:
                logLikelihood.value = logLikelihood.value + -(log(c_scaled[t]))
#                 total[t] = 0
#                 for j in range(1,N+1):
#                     total[t] += alpha_hat[(j,t)]*beta_hat[(j,t)]
#                 #total[t] computed here is P(X) or P_l
            #totalOverAllObservations[t] += total[t]
            for i in range(1, N + 1):
                gamma[(i,t)] = (alpha_hat[(i,t)]*beta_hat[(i,t)])/c_scaled[t]
                #totalAlphaBetaOverAllObeservations[(i,t)] += (alpha[(i,t)]*beta[(i,t)])
                #totalGamma[i] += gamma[(i,t)]
#                    totalGammaOverAllObservationsMinusLast[i] += gamma[(i,t)]
                with lock:
                    totalGammaOverAllObservations[i] += gamma[i, t]
#                    totalGammaDeltaOverAllObservations[(i,y[t-1])] += gamma[(i,t)]
                #totalGammaDeltaOverAllObservations_t[(i,y[t-1],x[i-1])] += gamma[(i,t)]
                #with totalGammaDeltaOverAllObservations_t.get_lock():
                with lock:
                    totalGammaDeltaOverAllObservations_t[i, indexMap[(y[t - 1], x[i - 1])]] = totalGammaDeltaOverAllObservations_t[i, indexMap[(y[t - 1], x[i - 1])]]+ gamma[i, t] #totalGammaDeltaOverAllObservations_t_overall_states[(y[t-1],x[i-1])] += gamma[(i,t)] #You can remove the upper variable
        
                #for j in range(1,N+1):
                #    xi[(i,j,t)] = (alpha[(i,t)]*a[(i,j)]*b[(j,y[t+1])]*beta[(j,t+1)])/total[t]
        t = T
        #             total[t] = 0.0
        #             for j in range(1,N+1):
        #                 total[t] += alpha[(j,t)]*beta[(j,t)]
        #Here value of total[T] is computed
        with lock:
            logLikelihood.value = logLikelihood.value + -(log(c_scaled[t]))
        for i in range(1, N + 1): #
            gamma[(i,t)] = (alpha_hat[(i,t)]*beta_hat[(i,t)])/c_scaled[t]
            with lock:
                totalGammaOverAllObservations[i] += gamma[i, t] #               totalGammaDeltaOverAllObservations[(i,y[t-1])] += gamma[(i,t)]
            #totalGammaDeltaOverAllObservations_t[(i,y[t-1],x[i-1])] += gamma[(i,t)]
            #with totalGammaDeltaOverAllObservations_t.get_lock():
            with lock:
                totalGammaDeltaOverAllObservations_t[i, indexMap[(y[t - 1], x[i - 1])]] = totalGammaDeltaOverAllObservations_t[i, indexMap[(y[t - 1], x[i - 1])]]+ gamma[i, t] #totalGammaDeltaOverAllObservations_t_overall_states[(y[t-1],x[i-1])] += gamma[(i,t)]
        print 'gamma',gamma
        for t in range(1, T):
            for i in range(1, N + 1):
                for j in range(1, N + 1):
#                        if iterations == 0:
                    xi[i, j, t] = alpha_hat[(i, t)] * a[(i, j)] * t_table[(y[t], x[j - 1])] * beta_hat[(j, t + 1)] #                        else:
        
#                            xi[(i,j,t)] = (alpha[(i,t)]*a[(i,j)]*b[(j,y[t])]*beta[(j,t+1)])/total[t]
#                        totalXiOverAllObservations[(i,j)] += xi[(i,j,t)]
        for i in range(1, N + 1):
            with lock:
                totalGamma1OverAllObservations[i] += gamma[i, 1]
        
        for d in range(-N - 1, N+1):
            #c[d] = 0
            for t in range(1, T):
                for i in range(1, N + 1):
                    if i + d <= N and i + d >= 1:
                        c[d] += xi[i, i + d, t]
        
        print 'c',c
        #Liang et al. suggestion
#             for d in c:
#                 if d < -7:
#                     c[-7] += c[d]
#                     c[d] = 0
#                 if d > 7:
#                     c[7] += c[d]
#                     c[d] = 0
        #print c
        for i in range(1, N + 1):
            for j in range(1, N + 1): #                     if j-i >= 7:
#                         totalC_j_Minus_iOverAllObservations[(i,j)] += c[7]
#                     elif j-i <= -7:
#                         totalC_j_Minus_iOverAllObservations[(i,j)] += c[-7]
#                     else:
                with lock:
                    totalC_j_Minus_iOverAllObservations[i, j] += c[j - i]
            
            for l in range(1, N + 1): #                     if l-i >= 7:
#                         totalC_l_Minus_iOverAllObservations[i] += c[7]
#                     elif l-i <= -7:
#                         totalC_l_Minus_iOverAllObservations[i] += c[-7]
#                     else:
                with totalC_l_Minus_iOverAllObservations.get_lock():
                    totalC_l_Minus_iOverAllObservations[i] += c[l - i]
    
#             for i in range(1,N+1):
#                 totalC[i] = 0
#                 for l in range(1,N+1):
#                     totalC[i] += c[l-i]
#        totalC = dict()
        print 'totalC_j_Minus_iOverAllObservations',totalC_j_Minus_iOverAllObservations
	print 'totalC_l_Minus_iOverAllObservations',totalC_l_Minus_iOverAllObservations 
    
    return N, i, a, pi, j


def parallelizeTComputationInMaximizationStep(N, biword, sd_size, totalGammaDeltaOverAllObservations_t, totalGammaDeltaOverAllObservations_t_overall_states,start,end):
    for k in range(start,end):
        for i in range(1, N + 1):
            totalGammaDeltaOverAllObservations_t_overall_states[k] += totalGammaDeltaOverAllObservations_t[i, k]
            (f,e) = biword[k]
            #totalGammaDeltaOverAllObservations_t_overall_states_over_dest[e] += totalGammaDeltaOverAllObservations_t[(i,k)]
    
    return i

def baumWelch(bitext_sd, s_count,t_table,sd_count):#L is the number of observations
    #set A,B,pi initially uniform at random
    #a = dict()
    #b = dict()
    #pi = dict()
    
    #For now, assume we know N ***************
    
    N = maxTargetSentenceLength(bitext_sd)
    print 'N',N
    N_max = N
    Y = bitext_sd
    #Y = initializeExample()
    yValues = s_count.keys()#computes all possible english words
    (indexMap,biword) = map_bitext_to_int(sd_count)
    sd_size = len(indexMap)
    #initializeUniformly(a,pi,N)
    
    #a = zeros((N+1,N+1))
    a_array = Array(ct.c_double,((N+1)*(N+1)))
    a_array2 = np.frombuffer(a_array.get_obj()) # mp_arr and arr share the same memory
    a = a_array2.reshape((N+1,N+1)) # b and arr share the same memory

    #pi = zeros(N+1)
    pi = Array('d', N+1)
    
    logLikelihood = Value('d',0.0)
    #logLikelihood = 0
    lastLogLikelihood = Value('d',0.0)
    #lastLogLikelihood = 0
    
    L = len(Y)
    #N = len(Y[0][1]) #first sentence x length
    #(a,pi) = initializeUniformly(N)
    
    #print yValues
    for iterations in range(0,10):
        startTime = time.time()
        print 'iteration',iterations
        logLikelihood.value = 0.0
        #logLikelihood = 0
        #E step
        #c = defaultdict(int)
        #totalGammaOverAllObservationsMinusLast = defaultdict(int)
        
        totalGammaOverAllObservations = Array('d', [0]*(N+1))
        #totalGammaOverAllObservations = Array('d', (N+1))
        #for i in range(N+1):
        #    totalGammaOverAllObservations[i] = 0
#        totalGammaDeltaOverAllObservations = defaultdict(int)
        #totalGammaDeltaOverAllObservations_t = defaultdict(int)
        #totalGammaDeltaOverAllObservations_t_array = Array(ct.c_double,((N+1)*(sd_size)))
        #totalGammaDeltaOverAllObservations_t_array2 = np.frombuffer(totalGammaDeltaOverAllObservations_t_array.get_obj()) # mp_arr and arr share the same memory
        #totalGammaDeltaOverAllObservations_t = totalGammaDeltaOverAllObservations_t_array2.reshape((N+1,sd_size)) # b and arr share the same memory
        totalGammaDeltaOverAllObservations_t = Array('d',[0]*((N+1)*(sd_size)))
        #totalGammaDeltaOverAllObservations_t = Array('d',((N+1)*(sd_size)))
        #amount = (N+1)*(sd_size)
        #for i in range(amount):
        #    totalGammaDeltaOverAllObservations_t[i] = 0
        #manager = mp.Manager()
        #totalGammaDeltaOverAllObservations_t = manager.dict()

	   #mp_arr2 = Array(c_double, m*k)
        totalGammaDeltaOverAllObservations_t_overall_states = Array('d',[0]*sd_size)
        #totalGammaDeltaOverAllObservations_t_overall_states = Array('d',sd_size)
        #for i in range(sd_size):
        #    totalGammaDeltaOverAllObservations_t_overall_states[i] = 0
        #totalGammaDeltaOverAllObservations_t_overall_states = zeros(sd_size)
#        totalXiOverAllObservations = defaultdict(int)
        #totalGamma1OverAllObservations = Array('d',N+1)
        totalGamma1OverAllObservations = Array('d',[0]*(N+1))
        #totalGamma1OverAllObservations = Array('d',(N+1))
        #for i in range(0,N+1):
        #    totalGamma1OverAllObservations[i] = 0.0
        
        totalC_j_Minus_iOverAllObservations_array = Array(ct.c_double,(N+1)*(N+1))
        totalC_j_Minus_iOverAllObservations_array2 = np.frombuffer(totalC_j_Minus_iOverAllObservations_array.get_obj())
        totalC_j_Minus_iOverAllObservations = totalC_j_Minus_iOverAllObservations_array2.reshape((N+1,N+1))
        for i in range(N+1):
            for j in range(N+1):
                totalC_j_Minus_iOverAllObservations[i,j] = 0.0
        #totalC_j_Minus_iOverAllObservations = Array('d',[0]*((N+1)*(N+1)))
        
        #totalC_j_Minus_iOverAllObservations = Array('d',((N+1)*(N+1)))
        #for i in range(0,(N+1)*(N+1)):
        #    totalC_j_Minus_iOverAllObservations[i] = 0.0
        
        #manager = mp.Manager()
        #totalC_j_Minus_iOverAllObservations = manager.dict()
#          
#         #mgr = mp.Manager()
        totalGammaDeltaOverAllObservations_t_overall_states_over_dest = defaultdict(int)
          
        #totalC_l_Minus_iOverAllObservations = Array('d',N+1)
        
        totalC_l_Minus_iOverAllObservations = Array('d',[0]*(N+1))
        
        #totalC_l_Minus_iOverAllObservations = Array('d',(N+1))
        #print totalC_l_Minus_iOverAllObservations[:]
        #for i in range(0,N+1):
        #    totalC_l_Minus_iOverAllObservations[i] = 0.0
        endTime = time.time()
        
        #print N
        print "%.2gs" % (endTime - startTime)
        logLikelihood.value = 0
#         totalGammaOverAllObservations = zeros(N+1)
# #        totalGammaDeltaOverAllObservations = defaultdict(int)
#         #totalGammaDeltaOverAllObservations_t = defaultdict(int)
#         totalGammaDeltaOverAllObservations_t = zeros((N+1,sd_size))
#         totalGammaDeltaOverAllObservations_t_overall_states = zeros(sd_size)
#         totalGammaDeltaOverAllObservations_t_overall_states_over_dest = defaultdict(int)
# #        totalXiOverAllObservations = defaultdict(int)
#         totalGamma1OverAllObservations = zeros(N+1)
#         totalC_j_Minus_iOverAllObservations = zeros((N+1,N+1))
#         totalC_l_Minus_iOverAllObservations = zeros(N+1)
 
        #for i in range(1,N+1):
        #        totalGamma[i] = 0
        #start l observations here
        #add forward backward as well
        intervals = 10
        jobs = []
        lock = mp.RLock()
        length_of_interval = L/intervals
        for i in range(0,intervals-1):
            start = i*length_of_interval
            end = (i+1)*length_of_interval
            #print start
            #print end
            p = Process(target=Expectation2, args = (lock, t_table, N, Y, sd_size, indexMap, iterations, totalGammaOverAllObservations, totalGammaDeltaOverAllObservations_t, totalGamma1OverAllObservations, totalC_j_Minus_iOverAllObservations, totalC_l_Minus_iOverAllObservations,start,end,a,pi,logLikelihood, lastLogLikelihood))
            p.start()
            jobs.append(p)
                      
        start = (intervals-1)*length_of_interval 
        end = L
        p = Process(target=Expectation2, args = (lock, t_table, N, Y, sd_size, indexMap, iterations, totalGammaOverAllObservations, totalGammaDeltaOverAllObservations_t, totalGamma1OverAllObservations, totalC_j_Minus_iOverAllObservations, totalC_l_Minus_iOverAllObservations,start,end,a,pi,logLikelihood, lastLogLikelihood))
        p.start()
        jobs.append(p)
        for p in jobs:
            p.join()

        
        endTime = time.time()
        
        #print N
        print "%.2gs" % (endTime - startTime)
        #N = len(totalGamma1OverAllObservations)-1
        #print N
        print 'last , new ', lastLogLikelihood.value, logLikelihood.value
        #print 'likelihood difference ', (logLikelihood.value - lastLogLikelihood.value)
        lastLogLikelihood.value = logLikelihood.value
        
        totalGammaOverAllObservationsOverAllStates = 0.0
	startTime = time.time()        
	for i in range(1,N+1):
            totalGammaOverAllObservationsOverAllStates += totalGammaOverAllObservations[i] 
            
            
            
        
        
#         for (i,f,e) in totalGammaDeltaOverAllObservations_t:
#             totalGammaDeltaOverAllObservations_t_overall_states[(f,e)] += totalGammaDeltaOverAllObservations_t[(i,f,e)]  
        a = zeros((N+1,N+1))
        pi = zeros(N+1)


        t_table = defaultdict(int)
       
        for i in range(1,N+1):
            pi[i] = totalGamma1OverAllObservations[i]*(1.0/L)
           
#             totalC[i] = 0
#             for l in range(1,N+1):
#                 totalC[i] += c[l-i]
#             for j in range(1,N+1):
#                 a[(i,j)] = c[j-i]/totalC[i] #transition with c*****
            for j in range(1,N+1):
                #address = i*(N+1) + j
                #a[(i,j)] = totalC_j_Minus_iOverAllObservations[address]/totalC_l_Minus_iOverAllObservations[i]
                a[(i,j)] = totalC_j_Minus_iOverAllObservations[(i,j)]/totalC_l_Minus_iOverAllObservations[i]
            #for j in range(1,N+1):
            #    a[(i,j)] = totalXiOverAllObservations[(i,j)]/totalGammaOverAllObservationsMinusLast[i]
                #transition standard fwbw
        startTime = time.time()
        for k in range(sd_size):
            for i in range(1, N + 1):
#FOR SLOW but correct dict version
#		if (i,k) in totalGammaDeltaOverAllObservations_t:
                address = i*sd_size + k
                totalGammaDeltaOverAllObservations_t_overall_states[k] += totalGammaDeltaOverAllObservations_t[address]
                (f,e) = biword[k]
                totalGammaDeltaOverAllObservations_t_overall_states_over_dest[e] += totalGammaDeltaOverAllObservations_t[address]
	
        endTime = time.time()
        for k in range(0,sd_size):
            f, e = biword[k]
            t_table[(f,e)] = totalGammaDeltaOverAllObservations_t_overall_states[k]/totalGammaDeltaOverAllObservations_t_overall_states_over_dest[e]
            #t_table[f, e] = totalGammaDeltaOverAllObservations_t_overall_states[k] / totalGammaOverAllObservationsOverAllStates
        endTime = time.time()
        
        #print N
        print "M time: %.2gs" %(endTime - startTime)

        del totalC_l_Minus_iOverAllObservations 
        
        
    return (a,t_table,pi)
    #print b
def viterbi(a,t_table,pi,N,o,d):
    V = dict()
    ptr = dict()
    #bPrime = defaultdict(int)
    #bPrime = b
    #print bPrime[(62,'on')]
    for q in range(1,N+1):
        if (o[0],d[q-1]) in t_table:
            V[(q,0)] = pi[q]*t_table[(o[0],d[q-1])]
        else:
            V[(q,0)] = 0
    for t in xrange(1,len(o)):
        for q in range(1,N+1):
            maximum = 0
            max_q = -1
            for q_prime in range(1,N+1):
                if (o[t],d[q-1]) in t_table:
                    if V[(q_prime,t-1)]*a[(q_prime,q)]*t_table[(o[t],d[q-1])] > maximum :
                        maximum = V[(q_prime,t-1)]*a[(q_prime,q)]*t_table[(o[t],d[q-1])]
                        max_q = q_prime
            V[q,t] = maximum
            ptr[q,t] = max_q 
    max_of_V = 0
    
    q_of_max_of_V = 0
    for q in range(1,N+1):
        #print 'viterbi value',V[(q,len(o)-1)]
        if V[(q,len(o)-1)] > max_of_V:
            max_of_V = V[(q,len(o)-1)]
            q_of_max_of_V = q
    #print max_of_V, q_of_max_of_V
    trace = [q_of_max_of_V]
    
    q = q_of_max_of_V
    i = len(o)-1
    
    while i > 0:
        q = ptr[(q,i)]   
        trace.insert(0, q)
        i = i -1
    return trace

def log_viterbi(a,t_table,pi,N,o,d):
    V = zeros((N+1,len(o)))
    ptr = zeros((N+1,len(o)))
    
    #bPrime = defaultdict(int)
    #bPrime = b
    #print bPrime[(62,'on')]
    for q in range(1,N+1):
        if (o[0],d[q-1]) in t_table:
            if t_table[(o[0],d[q-1])] == 0 or pi[q] == 0:
                V[(q,0)] = -sys.maxint
            else:
                V[(q,0)] = log(pi[q])+log(t_table[(o[0],d[q-1])])
        else:
            V[(q,0)] = 0
    for t in xrange(1,len(o)):
        for q in range(1,N+1):
            maximum = -sys.maxint
            max_q = -sys.maxint
            for q_prime in range(1,N+1):
                if (o[t],d[q-1]) in t_table:
                    if a[(q_prime,q)] != 0 and t_table[(o[t],d[q-1])] != 0 :
                        if V[(q_prime,t-1)]+log(a[(q_prime,q)])+log(t_table[(o[t],d[q-1])]) > maximum :
                            maximum = V[(q_prime,t-1)]+log(a[(q_prime,q)])+log(t_table[(o[t],d[q-1])])
                            max_q = q_prime
            V[q,t] = maximum
            ptr[q,t] = max_q 
    max_of_V = -sys.maxint
    
    q_of_max_of_V = 0
    for q in range(1,N+1):
        #print 'viterbi value',V[(q,len(o)-1)]
        if V[(q,len(o)-1)] > max_of_V:
            max_of_V = V[(q,len(o)-1)]
            q_of_max_of_V = q
    #print max_of_V, q_of_max_of_V
    trace = [q_of_max_of_V]
    
    q = q_of_max_of_V
    i = len(o)-1
    
    while i > 0:
        q = ptr[(q,i)]   
        trace.insert(0, q)
        i = i -1
    return trace




def findBestAlignmentsForAll(bitext,a,t_table,pi):
    for (n,(S,D)) in enumerate(bitext):
        
        N = len(D)
        bestAlignment = log_viterbi(a, t_table, pi, N, S, D)
        for (i,a_i) in enumerate(bestAlignment): 
            sys.stdout.write("%i-%i " % (i,a_i-1))
        sys.stdout.write("\n")
def findBestAlignmentsForAll_AER(bitext,a,t_table,pi,num_lines,alignmentFile):
    alignment = open(alignmentFile,'w')
    for (n,(S,D)) in enumerate(bitext):
        N = len(D)
        bestAlignment = log_viterbi(a, t_table, pi, N, S, D)
        for (i,a_i) in enumerate(bestAlignment):
            alignment.write("%i-%i " % (i+1,a_i))
	    #sys.stdout.write("%i-%i " % (i+1,a_i))
        alignment.write("\n")
	#sys.stdout.write('\n')
	if n == num_lines-1 :	
		return

def findBestAlignmentsForAllWithIntersection(bitext,a,b,pi,a_ds,b_ds,pi_ds):
    for (S,D) in bitext:
        set1 = set()
        set2 = set()
        N = len(D)
        N_ds = len(S)
        bestAlignment = log_viterbi(a, b, pi, N, S, D)
        bestAlignment_ds = log_viterbi(a_ds, b_ds, pi_ds, N_ds, D, S)
        for (i,a_i) in enumerate(bestAlignment): 
            set1.add((i,a_i-1))
        for (i,a_i) in enumerate(bestAlignment_ds): 
            set2.add((a_i-1,i))
        intersect = set1.intersection(set2)
        for (i_s,a_s) in intersect:
            sys.stdout.write("%i-%i " % (i_s,a_s))
        sys.stdout.write("\n")           
def findBestAlignmentsForAllWithIntersection_AER(bitext,a,b,pi,a_ds,b_ds,pi_ds,num_lines):
    alignment = open('alignment','w')
    for (n,(S,D)) in enumerate(bitext):
        set1 = set()
        set2 = set()
        N = len(D)
        N_ds = len(S)
        bestAlignment = log_viterbi(a, b, pi, N, S, D)
        bestAlignment_ds = log_viterbi(a_ds, b_ds, pi_ds, N_ds, D, S)
        for (i,a_i) in enumerate(bestAlignment): 
            set1.add((i,a_i-1))
        for (i,a_i) in enumerate(bestAlignment_ds): 
            set2.add((a_i-1,i))
        intersect = set1.intersection(set2)
        for (i_s,a_s) in intersect:
	    alignment.write("%i-%i " % (i_s+1,a_s+1))
            #sys.stdout.write("%i-%i " % (i_s,a_s))
        alignment.write('\n')
	if n == (num_lines - 1) :
		return
	#sys.stdout.write("\n") 

def readTable_sd(readFile):
    print 'firrst'
    t_sd = defaultdict(int)
    f = open(readFile, 'r')
    cnt = 0
    for line in f:
        l = line.split()
        s = l[0]
        d = l[1]
        p = l[2]
        t_sd[(s,d)] = float(p)
        if (cnt%100000) == 0 :
            sys.stderr.write(str(cnt))
            sys.stderr.write('\n')
        cnt += 1
    return t_sd 
def grade_align(reference, system, output):
    #f_data = "%s.%s" % (align_opts.train, align_opts.french)
    #e_data = "%s.%s" % (align_opts.train, align_opts.english)
    (size_a, size_s, size_a_and_s, size_a_and_p) = (0.0,0.0,0.0,0.0)
    for (n, (f, e, g, a)) in enumerate(zip(open(f_data), open(e_data), open(reference), open(system))):
      print n
      fwords = f.strip().split()
      ewords = e.strip().split()

      # check

      size_f = len(fwords)
      size_e = len(ewords)
      try:
        alignment = set([tuple(map(int, x.split("-"))) for x in a.strip().split()])
        for (i,j) in alignment:
          if (i>size_f or j>size_e):
            #print 'i',i,'j',j,'size_F',size_f,'size_e',size_e,'\n'
            sys.stderr.write("WARNING (%s): Sentence %d, point (%d,%d) is not a valid link\n" % (sys.argv[0],n,i,j))
          pass
      except (Exception):
        sys.stderr.write("ERROR (%s) line %d is not formatted correctly:\n  %s" % (sys.argv[0],n,a))
        sys.stderr.write("Lines can contain only tokens \"i-j\", where i and j are integer indexes into the French and English sentences, respectively.\n")
        sys.exit(1)

      # grade

      sure = set([tuple(map(int, x.split("-"))) for x in filter(lambda x: x.find("-") > -1, g.strip().split())])
      possible = set([tuple(map(int, x.split("?"))) for x in filter(lambda x: x.find("?") > -1, g.strip().split())])
      alignment = set([tuple(map(int, x.split("-"))) for x in a.strip().split()])
      size_a += len(alignment)
      size_s += len(sure)
      size_a_and_s += len(alignment & sure)
      size_a_and_p += len(alignment & possible) + len(alignment & sure)
#      if (i<opts.num_sents):
#        output.write("  Alignment %i  KEY: ( ) = guessed, * = sure, ? = possible\n" % i)
#        output.write("  ")
#        for j in ewords:
#          output.write("---")
#        output.write("\n")
#        for (i, f_i) in enumerate(fwords):
#          output.write(" |")
#          for (j, _) in enumerate(ewords):
#            (left,right) = ("(",")") if (i,j) in alignment else (" "," ")
#            point = "*" if (i,j) in sure else "?" if (i,j) in possible else " "
#            output.write("%s%s%s" % (left,point,right))
#          output.write(" | %s\n" % f_i)
#        output.write("  ")
#        for j in ewords:
#          output.write("---")
#        output.write("\n")
#        for k in range(max(map(len, ewords))):
#          output.write("  ")
#          for word in ewords:
#            letter = word[k] if len(word) > k else " "
#            output.write(" %s " % letter)
#          output.write("\n")
#        output.write("\n")

    precision = size_a_and_p / size_a
    recall = size_a_and_s / size_s
    aer = 1 - ((size_a_and_s + size_a_and_p) / (size_a + size_s))
    output.write("Precision = %f\nRecall = %f\nAER = %f\n" % (precision, recall, aer))
    print >>sys.stderr, "Precision = %f\nRecall = %f\nAER = %f\n" % (precision, recall, aer)
    return True
def writeParametersToFile(a,t_table,pi):
    aFile = open('a','wb')
    pFile = open('p','wb')
    tFile = open('t','wb')
    pickle.dump(t_table,tFile)
    tFile.close()
    pickle.dump(a,aFile)
    aFile.close()
    pickle.dump(pi,pFile)
    pFile.close()
def readParametersFromFile(aFileString, tFileString, pFileString):
	tFile = open(tFileString,'rb')
	t_table = pickle.load(tFile)
	#print t_table
	aFile = open(aFileString,'rb')
        a = pickle.load(aFile)
        #print a
	pFile = open(pFileString,'rb')
        pi = pickle.load(pFile)
        #print 'pi',pi
	return (a,t,pi)

def writeParameters(a,t_table,pi):
    aFile = open('a','w')
    pFile = open('p','w')
    tFile = open('t','w')
    N = a.shape[0]-1
    N2 = a.shape[1]-1
    print N,N2
    for i in range(1,N+1):
	for j in range(1, N+1):
	    aFile.write(str(a[(i,j)]) + ' ')
	aFile.write('\n')
    aFile.close()
    for i in range(1,N+1):
	pFile.write(str(pi[i])+'\t')
    for (f,e) in t_table :
    	tFile.write(e + ' ' + f + ' ' + str(t_table[(f,e)]) + '\n')

t_fe = EM_IBM1(f_count,e_count,fe_count,bitext_fe)
#t_fe = readTable_sd('/cs/natlang-user/amansour/t_fe.txt')
t_ef = EM_IBM1(e_count,f_count,ef_count,bitext_ef)
#start = time.clock()
(a,b,pi) = baumWelch(bitext_fe, f_count, t_fe,fe_count)
(a_ef,b_ef,pi_ef) = baumWelch(bitext_ef, e_count, t_ef,ef_count)
#end = time.clock()
#writeParametersToFile(a,b,pi)
#(a,b,pi) = readParametersFromFile('a_fe','t_fe','p_fe')
#print "%.2gs" % (end - start)

#findBestAlignmentsForAll_AER(bitext_test,a,b,pi,447,'alignment_fe')
#findBestAlignmentsForAllWithIntersection_AER(bitext_test, a, b, pi, a_ef, b_ef, pi_ef,447)
findBestAlignmentsForAllWithIntersection(bitext_test, a, b, pi, a_ef, b_ef, pi_ef)

#Computing AER
#alignment = "/cs/natlang-user/amansour/aligned_plus1" #alignment_fe_ef
#gold = "/cs/natlang-user/amansour/test.aligned.txt"

#output = sys.stdout
#grade_align(gold,alignment,output)




