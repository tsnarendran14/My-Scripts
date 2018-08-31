# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 13:22:21 2018

@author: narendran.thesma
"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities
import us
import pandas as pd
from collections import defaultdict
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_numeric, strip_multiple_whitespaces, strip_tags
import numpy as np
import re
import nltk
import itertools, collections
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import operator
from nltk.corpus import wordnet
import sys
import os
sys.path.insert(0, 'D:/Conagra/en/')

directory = "D:/Conagra/Events/Events/"
folders = os.listdir(directory)

finalEventScoreDF = pd.DataFrame()
for f in folders:
    try:
        tempDirectory = directory + f + "/"
        tempFiles = os.listdir(tempDirectory)
        tempMentionsFile = [n for n in tempFiles if 'Mentions' in n]
        tempDirectoryFile = tempDirectory + str(tempMentionsFile)
        tempDirectoryFile = tempDirectoryFile.replace("[","")
        tempDirectoryFile = tempDirectoryFile.replace("]","")
        tempDirectoryFile = tempDirectoryFile.replace("'","")
        #mention = pd.read_csv(tempDirectoryFile, skiprows = 9, encoding='ISO-8859-1')
        #mention = pd.read_csv("D:/Conagra/sysomos_download/BEDFORD_Aug-2017/disaster_07-01-17_BEDFORD/chk.csv", skiprows = 9, encoding='ISO-8859-1')  
        mention = pd.read_excel("D:/Conagra/sysomos_download/BEDFORD_Aug-2017/disaster_07-01-17_BEDFORD/chk.xlsx", skiprows = 7, encoding='ISO-8859-1')
        usCounties = pd.read_csv("D:/Conagra/UsCounties.csv")
        
        usCounties.Counties = usCounties.Counties.str.replace(" County","")
        usCounties = usCounties.Counties
        
        usCounties = ' '.join(w for w in usCounties)
        usCounties = usCounties.lower()
        usCounties = usCounties.split(" ")
        
        mentionContent = mention.Content.copy()
        
        mention.head()
        
        usStates = str(us.states.STATES)
        usStates = usStates.replace("<State:","")
        usStates = usStates.replace(">","")
        usStates = usStates.replace(",", "")
        usStates = usStates.replace("[", "")
        usStates = usStates.replace("]", "")
        usStates = usStates.lower()
        usStates = usStates.split()
        
        FmentionContent = pd.Series(len(mentionContent))
        
        for i in range(0, len(mentionContent)):
             FmentionContent[i] = remove_stopwords(mentionContent[i].lower())
             FmentionContent[i] = FmentionContent[i].replace("counties", "")
             FmentionContent[i] = FmentionContent[i].replace("county", "")
             FmentionContent[i] = FmentionContent[i].replace("state", "")          
             FmentionContent[i] = FmentionContent[i].replace("”", "")
             FmentionContent[i] = FmentionContent[i].replace("“", "")
             FmentionContent[i] = FmentionContent[i].replace("’", "")
             FmentionContent[i] = FmentionContent[i].replace("said", "")
             FmentionContent[i] = strip_numeric(FmentionContent[i])
             FmentionContent[i] = strip_tags(FmentionContent[i])
             FmentionContent[i] = strip_punctuation(FmentionContent[i])
             FmentionContent[i] = ' '.join([word for word in FmentionContent[i].split() if word not in usStates])
             FmentionContent[i] = ' '.join([word for word in FmentionContent[i].split() if word not in usCounties])
             FmentionContent[i] = ' '.join( [w for w in FmentionContent[i].split() if len(w)>3] )
             FmentionContent[i] = strip_multiple_whitespaces(FmentionContent[i])
             
        texts = FmentionContent.str.split()
        
        frequency = defaultdict(int)
        
        for text in texts:
            for token in text:
                frequency[token] += 1
        
        texts = [[token for token in text if frequency[token] > 1]for text in texts]
        
        dictionary = corpora.Dictionary(texts)
        
        print(dictionary)
        
        print(dictionary.token2id)
        
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        print(corpus)
        
        tfidf = models.TfidfModel(corpus)
        
        vec = [(0, 1), (4, 1)]
        
        print(tfidf[vec])
        
        corpus_tfidf = tfidf[corpus]
                
        for doc in corpus_tfidf:
            print(doc)
                
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=5)
        
        print(lsi.print_topics(num_topics=5, num_words=5))
        
        corpus_lsi = lsi[corpus_tfidf]
        
        lsi.print_topics(10)
        
        for doc in corpus_lsi:
            print(doc)
        
        index = similarities.SparseMatrixSimilarity(lsi[corpus_tfidf], num_features=100)
                
        sims = index[lsi[vec]]
        
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        simsDF = pd.DataFrame(sims)
        simsDF.columns = ['MentionID', 'LSIScore']
        simsDF.MentionID = simsDF.MentionID + 1
        
        ######################## BiGram ##################################################
        
        def ngrams(words, n=2, padding=False):
            "Compute n-grams with optional padding"
            pad = [] if not padding else [None]*(n-1)
            grams = pad + words + pad
            return (tuple(grams[i:i+n]) for i in range(0, len(grams) - (n - 1)))
        
        
        mentionSentences= FmentionContent
        
        words = list(word.lower() for mentionSentence in mentionSentences for word in mentionSentence.split(" "))
        for size, padding in ((3, 0), (4, 0), (2, 1)):
            print('\n%d-grams padding=%d' % (size, padding))
            print(list(ngrams(words, size, padding)))
        
        # show frequency
        counts = defaultdict(int)
        for ng in ngrams(words, 2, False):
            counts[ng] += 1
        
        FreqWordsDF = sorted(((c, ng) for ng, c in counts.items()), reverse=True)
        FreqWordsDF = pd.DataFrame(FreqWordsDF)
        
        FreqWordsDF.columns = ['Frequency', 'Words']
        
        FreqWordsDF = FreqWordsDF.loc[FreqWordsDF.Frequency > 2]
        
        FreqWordsDF.Words = FreqWordsDF['Words'].astype(str).str.replace("(", "")
        FreqWordsDF.Words = FreqWordsDF['Words'].astype(str).str.replace(")", "")
        FreqWordsDF.Words = FreqWordsDF['Words'].astype(str).str.replace(",", "")
        FreqWordsDF.Words = FreqWordsDF['Words'].astype(str).str.replace("'", "")
        
        FreqWordsDF['Words1'] = FreqWordsDF['Words'].astype(str).str.split(" ").str[0]
        FreqWordsDF['Words2'] = FreqWordsDF['Words'].astype(str).str.split(" ").str[1]
        
        for i in range(0, len(FmentionContent)):
            for j in range(0, len(FreqWordsDF)):
                if FreqWordsDF.Words[j] in FmentionContent[i]:
                    FmentionContent[i] = FmentionContent[i].replace(FreqWordsDF.Words[j], FreqWordsDF.Words1[j] + FreqWordsDF.Words2[j])
        
        ################# WordSimilarity ######################################
        
        usStates = str(us.states.STATES)
        usStates = usStates.replace("<State:","")
        usStates = usStates.replace(">","")
        usStates = usStates.replace(",", "")
        usStates = usStates.replace("[", "")
        usStates = usStates.replace("]", "")
        usStates = usStates.split()
        
        FmentionContentConcat = FmentionContent.str.cat(sep = ' ')
        FmentionContentConcat = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', FmentionContentConcat)
        FmentionContentConcat = strip_numeric(FmentionContentConcat)
        FmentionContentConcat = strip_punctuation(FmentionContentConcat)
        FmentionContentConcat = remove_stopwords(FmentionContentConcat)
        FmentionContentConcat = strip_multiple_whitespaces(FmentionContentConcat)
        
        for word in usStates:
            FmentionContentConcat = FmentionContentConcat.replace(word.lower(), "")
        FmentionContentConcat = strip_multiple_whitespaces(FmentionContentConcat)
        FmentionContentConcat = ' '.join( [w for w in FmentionContentConcat.split() if len(w)>3] )
                
        ######################## Word Count ###################################
        
        def word_count(str):
            counts = dict()
            words = str.split()
        
            for word in words:
                if word in counts:
                    counts[word] += 1
                else:
                    counts[word] = 1
        
            return counts
        
        ##################################################################
        
        FmentionContentConcat = word_count(FmentionContentConcat)
        
        sortedFmentionContentConcat = sorted(FmentionContentConcat.items(), key=operator.itemgetter(1), reverse=True)
        
        sortedFmentionContentConcatDF = pd.DataFrame(sortedFmentionContentConcat)
        
        mentionContentConcat = mentionContent.str.cat(sep = ' ')
        mentionContentConcat = nltk.word_tokenize(mentionContentConcat)
        result = nltk.pos_tag((mentionContentConcat))
        
        result = [i for i in result if i[1].lower() in ('nn')]
        
        resultDF =  pd.DataFrame(result)
        
        resultDF.columns = ['Word', 'Speech']
        resultDF = resultDF.drop_duplicates()
        sortedFmentionContentConcatDF.columns = ['Word', 'WordFrequency']
        
        nouncountDF = resultDF.merge(sortedFmentionContentConcatDF, on = 'Word'  ,how = 'left')
        
        nouncountDF = nouncountDF[np.isfinite(nouncountDF['WordFrequency'])]
        
        nounWordFrequencyDF = nouncountDF.sort_values('WordFrequency', ascending = False)
               
        ######################### Pairs #########################################################
        
        mentionSentences = FmentionContent.tolist()
               
        commonWords  =    [ 'a' ,'ability' ,'able' ,'about' ,'above' ,'accept' ,'according' ,'account' ,'across' ,'act' ,
                         'action' ,'activity' ,'actually' ,'add' ,'admit' ,'adult' ,'affect' , 'making', 'room', 'afternoon',
                         'after' ,'again' ,'against' ,'age','ago' ,'agree','ahead' ,'all' ,'allow' ,'almost' ,'alone' ,
                         'along' ,'already' ,'also' ,'although' ,'always' , 'army', 'public', 'private', 'weekend', 'week',
                         'American' , 'among' ,'amount','and' ,'another' ,'answer' ,'any' ,'anyone' ,'anything' ,'appear' ,
                         'apply' ,'approach' ,'area' ,'argue' ,'arm' ,'around' ,'arrive' ,'art' ,'article' ,'artist' ,
                         'as' ,'ask' ,'assume' ,'at' ,'attention' ,'available' ,'avoid' ,'away' ,'back' ,'bad' ,'be' , 'distance',
                         'beautiful' ,'because' ,'become' ,'before' ,'begin' ,'behind' ,'believe' ,'best' ,'better' ,
                         'between' ,'beyond' ,'big' ,'bit' ,'body' ,'both' ,'break' ,'bring' ,'brother' ,'build' , 'caught',
                         'building' , 'business' ,'but' ,'buy' ,'by' ,'call' ,'can' ,'candidate','care' ,'carry' ,'case' ,'catch' ,
                         'cause' ,'center' ,'central' ,'century' ,'certain' ,'certainly' ,'change' ,'character' , 'year',
                         'charge' , 'chief', 'check' ,'child' ,'count', 'county','choice' ,'choose' ,'citizen' ,'city' ,'civil' ,'class' ,'clear' ,
                         'clearly' ,'close' ,'cold' ,'collection' ,'color' ,'come' ,'common' ,'company' ,'compare' , 'cleanup',
                         'concern' ,'condition' ,'consider' ,'contain' ,'continue' ,'could' ,'country' ,'course' , 'unit', 'chief',
                         'cover' ,'create' ,'current' ,'cut', 'damage', 'dark' ,'daughter' ,'day' ,'decade' ,'decide' ,
                         'deep' ,'degree' , 'department', 'describe' ,'despite' ,'detail' ,'determine','die' ,'difference' ,
                         'different' ,'difficult' ,'dinner' ,'direction' ,'discuss' ,'discussion' ,'disease' ,'history', 'past',
                         'do' ,'door' ,'down' ,'draw' ,'drop' ,'during' ,'each' ,'early' ,'easy' ,'eat' ,'edge' ,'present',
                         'effort' ,'eight' ,'either' ,'else' ,'end' ,'enough' ,'enter' ,'entire' ,'especially' , 'photo',
                         'establish' ,'even' ,'evening' ,'ever' ,'every' ,'everybody' ,'everyone' ,'everything', 'community',
                         'exactly' ,'example' ,'exist' ,'expect' ,'experience' ,'explain' ,'eye' ,'face' ,'fact' , 'club', 'chance',
                         'factor' ,'fail' ,'fall' ,'family' ,'far' ,'fast' ,'father' ,'feel' ,'feeling' ,'few' , 'progress', 'game',
                         'field' ,'figure' ,'fill' ,'final' ,'finally','find' ,'fine' ,'finger' ,'finish' , 'firefighter',
                         'first','five' ,'floor' ,'fly' ,'focus' ,'follow','foot' ,'for' ,'forget' ,'form' ,'former' , 'percent'
                         'forward' ,'four' ,'free' ,'friend' ,'from' ,'front' ,'full','general' ,'get' ,'girl' ,'give' , 'percentage'
                         'go' ,'goal' ,'good' ,'great','group' ,'grow' ,'guess' ,'guy' ,'hair' ,'half' ,'hand' ,'hang' , 'lack'
                         'happen' ,'happy' ,'hard' ,'have' ,'he' ,'head' ,'hear' ,'heavy' ,'help' ,'her' ,'here' ,'herself' ,
                         'high' ,'him' ,'himself' ,'his' ,'hit' ,'hold' ,'home' ,'deal', 'hope' ,'hot' ,'hour' ,'how' ,'however' ,
                         'huge' ,'human' ,'hundred' ,'husband' ,'I' ,'idea','identify' ,'if' ,'image' ,'imagine' ,'impact' , 'http',
                         'important' ,'improve' ,'in' ,'include' ,'including' ,'increase' ,'indeed' ,'indicate' ,'individual' , 'com',
                         'information' ,'inside' ,'instead' ,'interest' ,'interesting' ,'international' ,'interview' ,'into' ,
                         'involve' ,'issue' ,'it' ,'item' ,'its' ,'itself' ,'job' ,'join' ,'just' ,'keep' ,'key' ,'kid' ,'kill' ,
                         'kind' ,'know' ,'knowledge' ,'land' ,'language' ,'large' ,'last' ,'late' ,'later' ,'laugh' ,'law' ,
                         'lay' ,'lead' ,'leader' ,'learn' ,'least' ,'leave' ,'left' ,'leg' ,'less' ,'let' ,'letter' ,'level' ,
                         'lie' ,'life' ,'light' ,'like' ,'likely' ,'line' ,'list' ,'listen' ,'little' ,'live' ,'local' ,'long' ,
                         'look' ,'lose' ,'loss' ,'lot' ,'look', 'love' ,'low' ,'main' ,'maintain' ,'major' ,'make' ,'man' ,'manage' ,
                         'management' ,'manager' ,'many' ,'matter' ,'may' ,'maybe' ,'me' ,'mean' ,'measure' ,'meet' ,'meeting' ,
                         'member' ,'mention' ,'message' ,'middle' ,'might' ,'million' ,'mind' ,'minute' ,'miss' ,'moment' ,
                         'month' ,'more' ,'morning' ,'most' ,'mother' ,'mouth', 'move' ,'movement' ,'Mr' ,'Mrs' ,'much' , 'driver',
                         'must' ,'my' ,'myself' ,'name' ,'nation', 'national' ,'near' ,'need','nearly' ,'necessary' ,'need' ,
                         'never' ,'new' ,'newspaper' ,'next' ,'nice' ,'no' ,'none' ,'nor' ,'north' ,'not' ,'note' ,'nothing' ,
                         'now' ,'number' ,'occur' ,'of' ,'off' ,'offer' ,'office' ,'officer' ,'official' ,'often' ,'oh' ,
                         'ok' ,'on' ,'once' ,'one' ,'only' ,'onto' ,'open' ,'opportunity' ,'option' ,'or' ,'order' ,'other' ,
                         'others' ,'our' ,'out' ,'outside' ,'over' ,'own' ,'owner' ,'page' ,'pain','part' ,'particular' ,
                         'particularly' ,'partner' ,'party' ,'pass' ,'past' ,'patient' ,'pay' ,'people' ,'per' ,'performance' ,
                         'perhaps' ,'period' ,'person' ,'phone' ,'pick' ,'picture' ,'piece' ,'place' ,'plan' ,'plant' ,'play' ,
                         'player' ,'PM' ,'point' , 'police', 'poor' ,'popular' ,'prepare' ,'present' ,'pretty' ,'prevent','price' ,'private' ,
                         'probably' ,'problem' ,'process' ,'produce' ,'product' ,'production' ,'professor' ,'program' ,'prove' ,
                         'provide' ,'pull' ,'purpose' ,'push' ,'put' ,'question' ,'quickly' ,'quite' ,'radio' ,'raise' ,'range' ,
                         'rate' ,'rather' ,'reach' ,'read' ,'ready' ,'real' ,'reality' ,'realize' ,'really', 'reason' ,'receive' ,
                         'recent' ,'recently' ,'recognize' ,'record' ,'red' ,'reduce','reflect' ,'region' ,'relate' ,'remain' ,
                         'remember' ,'remove' ,'report' ,'represent' ,'require','respond' ,'response' ,'responsibility' ,'rest' ,
                         'result' ,'return' ,'reveal' ,'rich' ,'right' ,'rise','role' ,'run' ,'same' ,'say' ,'scene' ,'score' , 'representative',
                         'sea' ,'season' ,'seat' ,'second' ,'section' ,'see' ,'seek' ,'seem' ,'sell' ,'send' ,'senior' ,'sense' , 'officer',
                         'series' ,'serious' ,'serve' ,'set' ,'seven' ,'several' ,'shake' ,'share' ,'she' ,'shoot' ,'short' , 'site', 'crew',
                         'shot' ,'should' ,'show' ,'side' ,'sign' ,'significant' ,'similar' ,'simple' ,'simply' ,'since' ,'sing' ,
                         'single' ,'sister' ,'sit' ,'site' ,'situation' ,'six' ,'size' ,'skill' ,'small' ,'smile' ,'so' ,'social' ,
                         'some' ,'somebody' ,'someone' ,'something' ,'sometimes' ,'son' ,'song' ,'soon' ,'sort' ,'sound' ,'source' ,
                         'south' ,'southern' ,'space' ,'speak' ,'special' ,'specific' ,'speech' ,'spend' ,'sport' ,'staff' ,
                         'stage' ,'stand' ,'standard' ,'star', 'said','start' ,'state' ,'statement' ,'stay', 'step' ,'still' ,'stop' ,
                         'store' ,'story' ,'street' ,'strong' ,'structure' ,'student' ,'study' ,'stuff' ,'style' ,'subject' , 'courtesy',
                         'success' ,'successful' ,'such' ,'suddenly' ,'suggest','support' ,'sure' ,'surface' ,'take' ,'talk' ,
                         'task','teach' ,'teacher' ,'team' ,'tell' ,'ten' ,'tend' ,'term' ,'test' ,'than' ,'thank' ,'that' ,
                         'the' ,'their', 'them' ,'themselves' ,'then' ,'there' , 'therere','these', 'thesere' ,'they' ,'theyre', 'thing' ,'think' ,'third' ,
                         'this' ,'those' ,'though' ,'thought' ,'thousand' ,'threat' ,'three' ,'through' ,'throughout' ,
                         'throw' ,'thus' ,'time' ,'to' ,'today' ,'together' ,'tonight' ,'too' ,'top' ,'total' ,'tough' ,
                         'toward' ,'town' ,'trade' ,'treat' ,'TRUE' ,'truth' ,'theme', 'try' ,'turn' ,'TV' ,'two' ,'type' ,'under' ,
                         'understand' ,'unit' ,'until' ,'up' ,'upon' ,'us' ,'use' ,'usually' ,'value' ,'various' ,'very' ,
                         'view' ,'visit' ,'voice' ,'vote' ,'watch' ,'wait' ,'walk' ,'wall' ,'want','way' ,'we' ,'wear' ,'week' ,'weight' ,
                         'well' ,'west' ,'western' ,'what' ,'whatever' ,'when' ,'where' ,'whether' ,'which' ,'while' ,'white' , 'questions',
                         'who' , 'went', 'whole' ,'whom' ,'whose' ,'why' ,'wide' ,'wife' ,'will' ,'win' ,'wind' ,'window' ,'wish' ,'with' ,
                         'within' ,'without' ,'woman' ,'wonder' ,'word' ,'work' ,'world' ,'worry' ,'would' ,'write' ,'wrong' ,
                         'yard' ,'yeah' ,'year', 'years' ,'yes' ,'yet' ,'you' ,'youre','young' ,'your' ,'yourself', 'yesterday']
        
        resultTag = {}
        for i in range(0,len(mentionContent)):
            ContentToken = nltk.word_tokenize(mentionContent[i])
            resultTag[i] = nltk.pos_tag(ContentToken)
        
        result = {}
        for i in range(0, len(resultTag)):
            temp = resultTag[i]
            result[i] = [i for i in temp if i[1].lower() in ('nn')]
        
        finalNoun = {}
        for i in range(0, len(result)):
            temp = result[i]
            noun = {}
            for j in range(0, len(temp)):
                noun[j] = temp[j][0]
            noun = str(noun)
            noun = noun.lower()
            finalNoun = str(finalNoun) + noun    
        
        finalNoun = strip_numeric(finalNoun)
        finalNoun = strip_punctuation(finalNoun)
        finalNoun = finalNoun.replace("“", "")
        finalNoun = finalNoun.replace("”", "")
        finalNoun = finalNoun.replace("’", "")
        finalNoun = strip_multiple_whitespaces(finalNoun)
        finalNoun = [w for w in finalNoun.split() if len(w)>3]
        finalNoun = [w for w in finalNoun if w not in commonWords]
        
        finalNoun = set(finalNoun)
        finalNoun = list(finalNoun)
                
        words = set(word.lower() for mentionSentence in mentionSentences for word in mentionSentence.split(" "))
        words = [w for w in words if w in finalNoun]
        words = [w for w in words if w not in commonWords]
        
        words = list(word.lower() for mentionSentence in mentionSentences for word in mentionSentence.split(" "))
        
        s = nltk.stem.snowball.EnglishStemmer()
        l = nltk.stem.WordNetLemmatizer()
        stemmedWordsDF = pd.DataFrame()
        for i in range(0, len(words)):
            tempStemmedWords = {"WordsOriginal": words[i], "StemmedWords" : s.stem(words[i]), "LemmatisedWords" : l.lemmatize(words[i])}
            tempStemmedWordsDF = pd.DataFrame([tempStemmedWords])
            stemmedWordsDF = stemmedWordsDF.append(pd.DataFrame(data = tempStemmedWordsDF))
        
        stemmedWordsDF = stemmedWordsDF.drop_duplicates()
        
        ############### Taking lemmatization - comment if you want to use stemming ######################
        
        stemmedWordsDF = stemmedWordsDF.drop(["StemmedWords"], axis = 1)
        stemmedWordsDF = stemmedWordsDF.drop_duplicates()
        stemmedWordsDF = stemmedWordsDF.rename(columns = {"LemmatisedWords" : "StemmedWords"})
        
        #########################################################################################
        
        words = stemmedWordsDF.StemmedWords
        words = [w for w in words if w in l.lemmatize(str(finalNoun))]
        words = [w for w in words if w not in l.lemmatize(str(commonWords))]
                
        words = set(words)
        
        mentionSentences = [[l.lemmatize(w) for w in sentence.split(" ")] for sentence in mentionSentences]
        
        for i in range(0, len(mentionSentences)):
            mentionSentences[i] = " ".join(mentionSentences[i])
        
        _pairs = list(itertools.permutations(words, 2))
        # We need to clean up similar pairs: sort words in each pair and then convert
        # them to tuple so we can convert whole list into set.
        pairs = set(map(tuple, map(sorted, _pairs)))
        
        c = collections.Counter()
        
        def exact_Match(phrase, word):
            b = r'(\s|^|$)' 
            return re.match(b + word + b, phrase, flags=re.IGNORECASE)
        
        pairDF = pd.DataFrame()
        i = 0
        for mentionSentence in mentionSentences:
            mentionSentence = mentionSentence.split(' ')
            for pair in pairs:
                #if re.match(r'\b' + pair[0] + r'\b', mentionSentence) and re.match(r'\b' + pair[1] + r'\b', mentionSentence):
                if pair[0] in mentionSentence and pair[1] in mentionSentence:
                #if pair[0] in mentionSentence and pair[1] in mentionSentence:
                    c.update({pair: 1})
                    temp = {'MentionID' : [i], 'Word1' : [pair[0]], 'Word2' : [pair[1]]}
                    tempDF = pd.DataFrame(data = temp)
                    pairDF = pairDF.append(tempDF)
            i = i + 1
        
        pairDFFrequency = pairDF.groupby(["Word1", "Word2"]).size().reset_index(name = "PairFrequency")
        
        pairDFFrequency = pairDFFrequency.sort_values(['PairFrequency'], ascending = False)
        
        pairDFFrequency = pairDFFrequency.merge(nounWordFrequencyDF, left_on = 'Word1', right_on = 'Word', how = 'inner' )
        
        pairDFFrequency = pairDFFrequency.rename(columns = {'WordFrequency' : 'WordFrequency1'})
        
        pairDFFrequency = pairDFFrequency.merge(nounWordFrequencyDF, left_on = 'Word2', right_on = 'Word', how = 'inner' )
        
        pairDFFrequency = pairDFFrequency.rename(columns = {'WordFrequency' : 'WordFrequency2'})
        
        pairDFFrequency = pairDFFrequency.drop('Word_x', axis = 1)
        pairDFFrequency = pairDFFrequency.drop('Word_y', axis = 1)
        pairDFFrequency = pairDFFrequency.drop('Speech_x', axis = 1)
        pairDFFrequency = pairDFFrequency.drop('Speech_y', axis = 1)
        
        finalPairDF = pairDF.merge(pairDFFrequency, left_on = ['Word1', 'Word2'], right_on = ['Word1', 'Word2'], how = 'inner')
        
        if(len(finalPairDF) > 0):
            finalPairDF = finalPairDF[finalPairDF.PairFrequency > 1 ]
                
        stemmedWordsDF = stemmedWordsDF.drop_duplicates()
        
        stemmedWordsOrginalTags = nltk.pos_tag(stemmedWordsDF.WordsOriginal)
        
        stemmedWordsOrginalTagsNoun = [item[0] for item in stemmedWordsOrginalTags if item[1] == "NN"]
        
        stemmedWordsDF = stemmedWordsDF[stemmedWordsDF['WordsOriginal'].isin(stemmedWordsOrginalTagsNoun)]
        
        stemUncommonWordsDF = stemmedWordsDF.loc[stemmedWordsDF['WordsOriginal'] != stemmedWordsDF['StemmedWords']]
        
        stemmedWordsDF = stemmedWordsDF.loc[~stemmedWordsDF['StemmedWords'].isin(list(stemUncommonWordsDF['StemmedWords']))]
        
        stemmedWordsDF = stemmedWordsDF.append(stemUncommonWordsDF)
        
        stemmedWordsDF = stemmedWordsDF.loc[~stemmedWordsDF['WordsOriginal'].isin(commonWords)]
                                
        finalPairDF = finalPairDF.merge(stemmedWordsDF, left_on = 'Word1', right_on = 'StemmedWords', how = 'inner' )
        finalPairDF = finalPairDF.rename(columns = {'WordsOriginal' : 'WordsOriginal1'})
        finalPairDF = finalPairDF.drop(['StemmedWords'], axis = 1)
        
        finalPairDF = finalPairDF.merge(stemmedWordsDF, left_on = 'Word2', right_on = 'StemmedWords', how = 'inner' )
        finalPairDF = finalPairDF.rename(columns = {'WordsOriginal' : 'WordsOriginal2'})
        finalPairDF = finalPairDF.drop(['StemmedWords'], axis = 1)
        
        finalPairDF['EventScore'] = 0.9 * finalPairDF['PairFrequency'] + 0.05 * finalPairDF['WordFrequency1'] + 0.05 * finalPairDF['WordFrequency2']
        
        EventsMentionIDDF = pd.DataFrame()
        
        finalPairDF.MentionID = finalPairDF.MentionID + 1
        for i in finalPairDF.MentionID.unique():
            temp = finalPairDF[finalPairDF.MentionID == i]
            tempWords = temp.Word1.append(temp.Word2)
            tempWords = tempWords.unique()
            finalSimilarityDictDF = pd.DataFrame()
            for j in range(0, len(tempWords)):
                try:
                   for k in  range(j + 1, len(tempWords)):
                      tempSimilarityDict = pd.DataFrame()
                      w1 = wordnet.synset(tempWords[j] + '.n' + '.01' )
                      w2 = wordnet.synset(tempWords[k] + '.n' + '.01' )
                      similarityQuotient = w1.wup_similarity(w2)
                      tempSimilarityDict = {'Word1': tempWords[j] , 'Word2':tempWords[k], 'similarityQuotient': similarityQuotient}
                      tempSimilarityDictDF = pd.DataFrame([tempSimilarityDict])
                      finalSimilarityDictDF = finalSimilarityDictDF.append(tempSimilarityDictDF)
                except:
                    print('Exception!')
            if(len(finalSimilarityDictDF) > 0):        
                finalSimilarityDictDF = finalSimilarityDictDF.loc[finalSimilarityDictDF['similarityQuotient'] < 0.5]
                if(len(finalSimilarityDictDF) > 0):
                    tempWords = finalSimilarityDictDF.Word1.append(finalSimilarityDictDF.Word2)
                    tempWords = tempWords.unique()
                    tempWords = ' '.join(tempWords)
                    tempDict = {'MentionID' : i, 'Events' : tempWords}
                    tempDF = pd.DataFrame([tempDict])
                    EventsMentionIDDF = EventsMentionIDDF.append(tempDF)   
                        
        simsPairDF = simsDF.merge(finalPairDF, left_on = 'MentionID', right_on = 'MentionID', how = 'left')
        
        highLSIPair = simsPairDF.loc[simsPairDF['LSIScore'] >= 0.7]
        
        highLSIPair = highLSIPair.sort_values(['PairFrequency'], ascending=[False])
        
        if len(highLSIPair) > 0:
            highLSIEventsDF = pd.DataFrame()
            for i in range(0, len(highLSIPair)):
                tempDF = highLSIPair.iloc[[i]]
                tempWords = tempDF.WordsOriginal1.append(tempDF.WordsOriginal2)
                tempWordsDF = highLSIPair.loc[highLSIPair['WordsOriginal1'].isin(tempDF.WordsOriginal1) & highLSIPair['WordsOriginal2'].isin(tempDF.WordsOriginal2)]
                if len(tempWordsDF.MentionID.unique()) > 1:
                    temphighLSIEvents = {"WordOriginal1" : tempDF.WordsOriginal1, "WordOriginal2" :  tempDF.WordsOriginal2, 'WordFrequency1' : tempDF.WordFrequency1, 'WordFrequency2' : tempDF.WordFrequency2, 'NumberOfMentionSentences' : len(tempWordsDF.MentionID.unique())}
                    temphighLSIEventsDF = pd.DataFrame(temphighLSIEvents)
                    highLSIEventsDF = highLSIEventsDF.append(temphighLSIEventsDF)
                    highLSIEventsDF = highLSIEventsDF.drop_duplicates()
        
        highLSIEventsSimilarityDF = pd.DataFrame()
        if len(highLSIEventsDF) > 0:
            for i in range(0, len(highLSIEventsDF)):
                tempDF = highLSIEventsDF.iloc[i]
                try:
                    w1 = wordnet.synset(tempDF.WordOriginal1 + '.n' + '.01' )
                    w2 = wordnet.synset(tempDF.WordOriginal2 + '.n' + '.01' )
                    tempDF['similarityQuotient'] = w1.wup_similarity(w2)
                    tempDict = {'NumberOfMentionSentences' : tempDF['NumberOfMentionSentences'], 'WordFrequency1' : tempDF['WordFrequency1'], 'WordFrequency2' : tempDF['WordFrequency2'], 'WordOriginal1' : tempDF['WordOriginal1'], 'WordOriginal2' : tempDF['WordOriginal2'], 'similarityQuotient' : tempDF['similarityQuotient']}
                    tempDictDF = pd.DataFrame([tempDict])
                    highLSIEventsSimilarityDF = highLSIEventsSimilarityDF.append(tempDictDF)
                except:
                    print('')
                        
        if len(highLSIEventsSimilarityDF) > 0:
            highLSIEventsSimilarityDF = highLSIEventsSimilarityDF.loc[highLSIEventsSimilarityDF['similarityQuotient'] < 0.5]        
            if len(highLSIEventsSimilarityDF.NumberOfMentionSentences.unique()) > 0 :    
                tempDF = highLSIEventsSimilarityDF.loc[highLSIEventsSimilarityDF['NumberOfMentionSentences'] == max(highLSIEventsSimilarityDF.NumberOfMentionSentences)]
                tempDF['WordFrequencySum'] = tempDF['WordFrequency1'] + tempDF['WordFrequency2']
                tempDF = tempDF.loc[tempDF['WordFrequencySum'] == max(tempDF.WordFrequencySum)]
                tempDF = tempDF.head(1)
                majorHighEvents = tempDF.WordOriginal1.append(tempDF.WordOriginal2)
                majorHighEvents = ' '.join(majorHighEvents)
            simDFHigh = simsDF.loc[simsDF['LSIScore'] >= 0.7]
            simDFHigh['Events'] = majorHighEvents
            wordsToBeIgnored = majorHighEvents.split(' ')
                
            if len(simDFHigh) > 0:
                lowLSIPairAll = simsPairDF.loc[simsPairDF['LSIScore'] < 0.7]
                
                lowLSIPairAll = lowLSIPairAll.loc[~(lowLSIPairAll['WordsOriginal1'] == wordsToBeIgnored[0]) & ~(lowLSIPairAll['WordsOriginal2'] == wordsToBeIgnored[1])]
                lowLSIPairAll = lowLSIPairAll.loc[~(lowLSIPairAll['WordsOriginal2'] == wordsToBeIgnored[0]) & ~(lowLSIPairAll['WordsOriginal1'] == wordsToBeIgnored[1])]
                simsDFLowAll = pd.DataFrame()
                for z in lowLSIPairAll.MentionID.unique():
                    try:
                        lowLSIPair = lowLSIPairAll.loc[lowLSIPairAll['MentionID'] == z]
                        lowLSIEventsDF = pd.DataFrame()
                        for i in range(0, len(lowLSIPair)):
                            tempDF = lowLSIPair.iloc[[i]]
                            tempWords = tempDF.WordsOriginal1.append(tempDF.WordsOriginal2)
                            tempWordsDF = lowLSIPair.loc[lowLSIPair['WordsOriginal1'].isin(tempDF.WordsOriginal1) & lowLSIPair['WordsOriginal2'].isin(tempDF.WordsOriginal2)]
                            if len(tempWordsDF.MentionID.unique()) > 0:
                                templowLSIEvents = {"WordOriginal1" : tempDF.WordsOriginal1, "WordOriginal2" :  tempDF.WordsOriginal2, 'WordFrequency1' : tempDF.WordFrequency1, 'WordFrequency2' : tempDF.WordFrequency2, 'NumberOfMentionSentences' : len(tempWordsDF.MentionID.unique())}
                                templowLSIEventsDF = pd.DataFrame(templowLSIEvents)
                                lowLSIEventsDF = lowLSIEventsDF.append(templowLSIEventsDF)
                                lowLSIEventsDF = lowLSIEventsDF.drop_duplicates()    
                        lowLSIEventsSimilarityDF = pd.DataFrame()
                        for j in range(0, len(lowLSIEventsDF)):
                            tempDF = lowLSIEventsDF.iloc[j]
                            try:
                                w1 = []
                                w2 = []
                                w1 = wordnet.synset(tempDF.WordOriginal1 + '.n' + '.01' )
                                w2 = wordnet.synset(tempDF.WordOriginal2 + '.n' + '.01' )
                                tempDF['similarityQuotient'] = w1.wup_similarity(w2)
                                tempDict = {'NumberOfMentionSentences' : tempDF['NumberOfMentionSentences'], 'WordFrequency1' : tempDF['WordFrequency1'], 'WordFrequency2' : tempDF['WordFrequency2'], 'WordOriginal1' : tempDF['WordOriginal1'], 'WordOriginal2' : tempDF['WordOriginal2'], 'similarityQuotient' : tempDF['similarityQuotient']}
                                tempDictDF = pd.DataFrame([tempDict])
                                lowLSIEventsSimilarityDF = lowLSIEventsSimilarityDF.append(tempDictDF)
                            except:
                                print('Exception')                
                        lowLSIEventsSimilarityDF = lowLSIEventsSimilarityDF.loc[lowLSIEventsSimilarityDF['similarityQuotient'] < 0.5]        
                        if len(lowLSIEventsSimilarityDF.NumberOfMentionSentences.unique()) > 0 :    
                            tempDF = lowLSIEventsSimilarityDF.loc[lowLSIEventsSimilarityDF['NumberOfMentionSentences'] == max(lowLSIEventsSimilarityDF.NumberOfMentionSentences)]
                            tempDF['WordFrequencySum'] = tempDF['WordFrequency1'] + tempDF['WordFrequency2']
                            tempDF = tempDF.loc[tempDF['WordFrequencySum'] == max(tempDF.WordFrequencySum)]
                            tempDF = tempDF.head(1)
                            majorLowEvents = tempDF.WordOriginal1.append(tempDF.WordOriginal2)
                            majorLowEvents = ' '.join(majorLowEvents)
                        
                        simDFLow = {'MentionID':z, 'LSIScore':lowLSIPair.LSIScore.unique(), 'Events' : majorLowEvents }
                        simDFLow = pd.DataFrame(simDFLow)
                        simsDFLowAll = simsDFLowAll.append(simDFLow)
                    except:
                        print('Error Main Loop')
                                        
                finalPairDFCopy = finalPairDF.copy()       
                EventScoreMentionIDDF = finalPairDFCopy.loc[:,['MentionID', 'EventScore']]
                
                EventScoreMentionIDDF = EventScoreMentionIDDF.groupby(['MentionID']).sum()
                
                simDFHigh['Score'] = 5
                
                simsDFLowAll['Score'] = np.where(simsDFLowAll['LSIScore'] >= simsDFLowAll['LSIScore'].quantile(.75), 4, np.where((simsDFLowAll['LSIScore'] < simsDFLowAll['LSIScore'].quantile(.75)) & (simsDFLowAll['LSIScore'] >= simsDFLowAll['LSIScore'].quantile(.5)) ,3 , np.where((simsDFLowAll['LSIScore'] >= simsDFLowAll['LSIScore'].quantile(.25)) & (simsDFLowAll['LSIScore'] < simsDFLowAll['LSIScore'].quantile(.5)) , 2, 1)))
                
                simDFEventsLSIID = simDFHigh.append(simsDFLowAll)
                
                EventsMaxScoreDF = pd.DataFrame()
                for i in simDFEventsLSIID.Events.unique():
                    tempDF = simDFEventsLSIID.loc[simDFEventsLSIID['Events'] == i]
                    tempDict = {'Events' : i, 'MaxScore' : max(tempDF['Score'])}
                    tempFinalDF = pd.DataFrame([tempDict])
                    EventsMaxScoreDF = EventsMaxScoreDF.append(tempFinalDF)
                
                simDFEventsLSIID = simDFEventsLSIID.merge(EventsMaxScoreDF, left_on = 'Events', right_on = 'Events', how = 'left')
                
        ########################### Sentiment ############################################
        
        sid = SentimentIntensityAnalyzer()
        
        finalPolarityScore = pd.DataFrame()
        for i in range(0, len(mentionContent)):
            tempMentionContent = mentionContent[i]
            tempPolarityScore = sid.polarity_scores(tempMentionContent)
            tempPolarityScore.update({'MentionID' : i })
            tempPolarityScoreDF = pd.DataFrame([tempPolarityScore])
            finalPolarityScore = finalPolarityScore.append(tempPolarityScoreDF)
            
        finalPolarityScore['MentionID'] = finalPolarityScore['MentionID'] + 1
        
        if len(simDFEventsLSIID) > 0:
            finalSentimentEventDF = simDFEventsLSIID.merge(finalPolarityScore, left_on = 'MentionID', right_on = 'MentionID', how = 'left')
            
            mentionCopy = mention.copy()
            
            mentionCopy = mentionCopy.loc[:,['Date (ET)', 'Title']]
            mentionCopy['Date (ET)'] = pd.to_datetime(mentionCopy['Date (ET)'])
            mentionCopy['Month'] = mentionCopy['Date (ET)'].dt.strftime("%B")
            mentionCopy['MentionID'] = list(range(1, len(mentionCopy) + 1))
            mentionCopy = mentionCopy.drop(['Date (ET)'], axis = 1)
            
            if len(finalSentimentEventDF) > 0:
                finalOutputDF = finalSentimentEventDF.merge(mentionCopy, left_on = 'MentionID', right_on = 'MentionID', how='left')
                finalEventScoreDF = finalEventScoreDF.append(finalOutputDF)

    except:
        print('Exception Folder')
        
finalEventScoreDF = finalEventScoreDF.dropna(axis = 0, how = 'any')
