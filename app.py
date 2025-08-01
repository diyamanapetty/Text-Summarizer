# -*- coding: utf-8 -*-
import streamlit as st
import spacy
from collections import Counter
from spacy.matcher import Matcher
from textblob import TextBlob
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import matplotlib.pyplot as plt

# Load the NLP model
nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

# Set up the page
st.set_page_config(page_title=" Text Summarization System using Fuzzy", layout="wide")
st.title(" Text Summarization System")
st.write("This system ranks sentences based on similarity, sentiment, and POS tags.")

# Initialize session state for inputs

if 'text_document' not in st.session_state:
    st.session_state.text_document = "There is a developer beautiful and great conference happening on 21 July 2019 in London. The conference is in area of biological sciences."
if 'sentence_to_rank' not in st.session_state:
    st.session_state.sentence_to_rank = "Financially good conference situations."

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Document")
    text_document = st.text_area("Enter the document text:",
                                 key="text_document",
                                 height=200)

with col2:
    st.subheader("Sentence to Rank")
    sentence_to_rank = st.text_area("Enter the sentence to rank:",
                                    key="sentence_to_rank",
                                    height=100)



# Fuzzy logic setup
def setup_fuzzy_system():
    # Antecedents
    similarity_document = ctrl.Antecedent(np.arange(0, 1.25, 0.1), 'similarity')
    sentiment_score = ctrl.Antecedent(np.arange(0, 1.25, 0.1), 'sentiment_score')
    nounCount = ctrl.Antecedent(np.arange(0, 110, 10), 'nounCount')
    verbCount = ctrl.Antecedent(np.arange(0, 110, 10), 'verbCount')
    
    # Consequent
    rank = ctrl.Consequent(np.arange(0, 24, 1), 'rank')

    # Membership functions for similarity
    similarity_document['low'] = fuzz.trimf(similarity_document.universe, [0, 0.3, 0.5])
    similarity_document['average'] = fuzz.trimf(similarity_document.universe, [0.3, 0.7, 1])
    similarity_document['high'] = fuzz.trimf(similarity_document.universe, [0.7, 1, 1.25])

    # Membership functions for sentiment
    sentiment_score['low'] = fuzz.trimf(sentiment_score.universe, [0, 0.3, 0.5])
    sentiment_score['average'] = fuzz.trimf(sentiment_score.universe, [0.3, 0.7, 1])
    sentiment_score['high'] = fuzz.trimf(sentiment_score.universe, [0.7, 1, 1.25])

    # Membership functions for noun count
    nounCount['low'] = fuzz.trimf(nounCount.universe, [0, 30, 50])
    nounCount['average'] = fuzz.trimf(nounCount.universe, [30, 70, 100])
    nounCount['high'] = fuzz.trimf(nounCount.universe, [70, 100, 110])

    # Membership functions for verb count
    verbCount['low'] = fuzz.trimf(verbCount.universe, [0, 30, 50])
    verbCount['average'] = fuzz.trimf(verbCount.universe, [30, 50, 70])
    verbCount['high'] = fuzz.trimf(verbCount.universe, [50, 70, 110])

    # Membership functions for rank
    rank['low'] = fuzz.trimf(rank.universe, [0, 0, 10])
    rank['average'] = fuzz.trimf(rank.universe, [10, 12, 24])
    rank['high'] = fuzz.trapmf(rank.universe, [12, 18, 24, 24])

    # Rules
    rule1 = ctrl.Rule(similarity_document['low'] | sentiment_score['low'], rank['low'])
    rule2 = ctrl.Rule(sentiment_score['average'], rank['average'])
    rule3 = ctrl.Rule(sentiment_score['average'] | similarity_document['average'], rank['average'])
    rule10 = ctrl.Rule(similarity_document['high'] & nounCount["high"] & sentiment_score['high'] & verbCount["high"], rank['high'])

    rank_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule10])
    return ctrl.ControlSystemSimulation(rank_ctrl), rank  # Return both the simulation and rank objects

# Functions from original code
def getMainNounChunk(inputSentence):
    lenChunk = 0
    prevLen = -1
    mainChunk = ""
    for chunk in inputSentence.noun_chunks:
        lenChunk = len(chunk)
        if prevLen < lenChunk:
            mainChunk = chunk
            prevLen = lenChunk
    return mainChunk

def getsentiment2(sent):
    sentimentObject = TextBlob(sent.text)
    sentimentObject = sentimentObject.sentiment
    return sentimentObject.polarity * 10

def getSimilarity(sentence1, doc1):
    return doc1.similarity(sentence1)

def getPOSCOUNT(inputText, inputPOSTag):
    countPOS = 0
    doc = nlp(inputText)
    for token in doc:
        countPOS += 1
        dictonaryInputText = Counter([token.pos_ for token in doc])
    return dictonaryInputText[inputPOSTag]/(countPOS+1) * 100

# Process button
if st.button("Rank Sentence"):
    with st.spinner("Processing..."):
        # Process the inputs
        doc = nlp(text_document)
        sent = nlp(sentence_to_rank)
        
        # Get all metrics
        main_chunk = getMainNounChunk(sent)
        similarity = getSimilarity(sent, doc)
        sentiment = getsentiment2(sent)
        noun_count = getPOSCOUNT(text_document, "NOUN")
        verb_count = getPOSCOUNT(text_document, "VERB")
        
        # Set up and run fuzzy system
        rankFIS, rank_obj = setup_fuzzy_system()  # Get both the simulation and rank objects
        rankFIS.input['similarity'] = similarity
        rankFIS.input['sentiment_score'] = sentiment
        rankFIS.input['nounCount'] = noun_count
        rankFIS.input['verbCount'] = verb_count
        rankFIS.compute()
        rank_score = rankFIS.output['rank']
        
        # Display results
        st.subheader("Results")
        
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric("Main Noun Chunk", str(main_chunk))
            st.metric("Similarity Score", f"{similarity:.2f}")
            st.metric("Sentiment Score", f"{sentiment:.2f}")
            
        with metrics_col2:
            st.metric("Noun Percentage", f"{noun_count:.2f}%")
            st.metric("Verb Percentage", f"{verb_count:.2f}%")
        
        st.success(f"Final Rank Score: {rank_score:.2f}")

if st.button("Summarize Document"):
    with st.spinner("Summarizing..."):
        doc = nlp(text_document)
        sentence_scores = []

        for sentence in doc.sents:
            try:
                # Process sentence
                main_chunk = getMainNounChunk(sentence)
                similarity = getSimilarity(sentence, doc)
                sentiment = getsentiment2(sentence)
                noun_count = getPOSCOUNT(sentence.text, "NOUN")
                verb_count = getPOSCOUNT(sentence.text, "VERB")

                # Setup fuzzy system
                rankFIS, rank_obj = setup_fuzzy_system()
                rankFIS.input['similarity'] = float(similarity)
                rankFIS.input['sentiment_score'] = float(sentiment)
                rankFIS.input['nounCount'] = float(noun_count)
                rankFIS.input['verbCount'] = float(verb_count)

                rankFIS.compute()  # << MUST call this before accessing output
                rank = rankFIS.output['rank']

                sentence_scores.append((sentence.text.strip(), rank))
            except Exception as e:
                st.warning(f"Skipping a sentence due to error: {e}")

        if sentence_scores:
            sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
            top_sentences = [sent[0] for sent in sorted_sentences[:3]]

            st.subheader("Extracted Summary")
            for i, s in enumerate(top_sentences, 1):
                st.markdown(f"**{i}.** {s}")
        else:
            st.error("No valid sentences could be summarized.")

        
# Sidebar with information
with st.sidebar:
    st.header("About")
    st.write("""
    This system ranks sentences based on:
    - Similarity to document
    - Sentiment analysis
    - Part-of-speech distribution
    - Fuzzy logic rules
    """)
    
    st.header("Instructions")
    st.write("""
    1. Enter your document text in the left box
    2. Enter the sentence you want to rank in the right box
    3. Click the 'Rank Sentence' button
    4. View the results and metrics
    """)
    
    st.header("Requirements")
    st.code("""
    pip install streamlit spacy textblob scikit-fuzzy matplotlib
    python -m spacy download en_core_web_sm
    """)