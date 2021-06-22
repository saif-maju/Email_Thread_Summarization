{{page.authors}}

Deep learning models have tons of applications in real world. They have reduced a lot of human effort through minimizing time consuming and repetative tasks. Modern deep learning models do not only offer efficient results, but many provide multi-purpose capacility too. 

 This article discusses email thread summarization through a unified Text-To-Text Transfer Transformer (T5) put forward by Google in a paper called "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" 
 
 

![T5](https://1.bp.blogspot.com/-o4oiOExxq1s/Xk26XPC3haI/AAAAAAAAFU8/NBlvOWB84L0PTYy9TzZBaLf6fwPGJTR0QCLcBGAsYHQ/s640/image3.gif)


# Table of Contents

1. Text Summarization 
    - Extractive Summary 
    - Abstractive Summary
2. 
2. Dataset Description
    - Colossal Clean Crawled Corpus (C4)
    - BC3 Email Corpus   


# 1. Text Summarization
Humans can easily summarize any text, such as an essay or article, by just reading it, noting or highlighting key points and then parapharsing it in their own words. This seems pretty simple tasks to do. However, machine cannot learn so easily. People tried different methods over time to make it learn how to summrize any text. Two of the major approaches still used are described below.  

## Extractive Summarization
With this approach, we use the firststp from a human course of action: Highlighting the important points. In this technique, we select most informative senetences from the text and use them as a summary. This is a naive approach but in most cases it produces very good results as often some sentences provided as much information as necessary to understand the whole text.

We have applied this technique using TextRank Alogorithm which we will cover in the Modeling section.

## Abstractive Summarization
This approach is more humanly as it not only remembers key points as well as reproduces them in natural language. It is a difficult task and requires complex language modeling. This is why we use state-of-the-art T5 model to model this. 
This will be the focus of our article. 

# 2. Dataset Description

## Colossal Clean Crawled Corpus (C4)

This dataset is a shortened and cleaned version of [Common Crawl Dataset](https://commoncrawl.org/). It was reduced from 20TB to just 75GB by apolying some heuristics to clean the text. It is web extracted text which contained content in HTML pages, with markup and non-text content removed. Google as used data from April 2019 for pre-training T5 and released it as part of TensorFlow Datasets.

## BC3 Email Corpus 
This [dataset](https://www.cs.ubc.ca/cs-research/lci/research-groups/natural-language-processing/bc3.html) is a collection of 40 email threads (3222 sentences) from W3C corpus. This is prepared for summarization task, annotated by three different annotators. They provided both extractive and abstractive summaries which could be used for training the model as a target summary. A number of researchers have used this dataset for serveral NLP tasks.

We will use this dataset for fine-tuning T5. Another section of this article will discuss preprocessing of this dataset. 
