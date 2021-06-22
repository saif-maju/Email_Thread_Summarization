
Deep learning models have tons of applications in real world. They have reduced a lot of human effort through minimizing time consuming and repetative tasks. Modern deep learning models do not only offer efficient results, but many provide multi-purpose capacility too. 

 This article discusses email thread summarization through a unified Text-To-Text Transfer Transformer (T5) put forward by Google in a paper called "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" 
 
 

![T5](https://1.bp.blogspot.com/-o4oiOExxq1s/Xk26XPC3haI/AAAAAAAAFU8/NBlvOWB84L0PTYy9TzZBaLf6fwPGJTR0QCLcBGAsYHQ/s640/image3.gif)

   *Unified Framework for All Downstream Tasks via Google AI Blog*
# Table of Contents

1. Text Summarization 
    - Extractive Summary 
    - Abstractive Summary
2. Transfer Learning and Transformer Model
    
    *To be written*
3. Dataset Description
    - Colossal Clean Crawled Corpus (C4)
    - British Columbia Conversation Corpora (BC3)   
4. Data Extraction and Preprocessing


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

This dataset is a shortened and cleaned version of [Common Crawl Dataset](https://commoncrawl.org/). It was reduced from 20TB to just 75GB by apolying some heuristics to clean the text. It is web extracted text which contained content in HTML pages, with markup and non-text content removed. Google as used data from April 2019 for pre-training T5 and released it as part of [TensorFlow Datasets](https://www.tensorflow.org/datasets/).

## British Columbia Conversation Corpora (BC3)
This [dataset](https://www.cs.ubc.ca/cs-research/lci/research-groups/natural-language-processing/bc3.html) is a collection of 40 email threads (3222 sentences) from W3C corpus. This is prepared for summarization task, annotated by three different annotators. They provided both extractive and abstractive summaries which could be used for training the model as a target summary. A number of researchers have used this dataset for serveral NLP tasks.

We will use this dataset for fine-tuning T5. Another section of this article will discuss preprocessing of this dataset. 

# 4. Data Extraction and Preprocessing

The dataset can be downloaded from website of [The University of British Columbia](https://www.cs.ubc.ca/cs-research/lci/research-groups/natural-language-processing/bc3.html). It is also available on our [GitHub Repo](https://github.com/saif-maju/Email_Thread_Summarization). Once downloaded, we can preprocess it in Python. 

This dataset is split into two XML files. One contains the original emails split line by line, and the other contains the summaries created by the annotators. Each email may contain several summaraies from different annotators and summaries may also be over several emails. 

Since emails in XML format, we need to parse them. For this we will use ElementTree package, that has functions to read and manipulate XMLs. 

First, import ElementTree. It's a common practice to use the alias of ET:

```
import xml.etree.ElementTree as ET
```
Then you need to read in the file with ElementTree.

```
parsedXML = ET.parse( "/BC2_Email_Corpus/corpus.xml" )
root = parsedXML.getroot()
```


```
bc3_email_df = parse_bc3_emails(root)
bc3_email_df.head(2)
```
Output: 

|listno	| date  |	from  |	to |	subject	| body |	email_num |
| ---   | ---   | --- | --- | --- | --- | --- |
|0 | 007-7484738 | Tue, 08 Dec 1998 07:30:52 -0800 |	Jacob Palme	| discuss@apps.ietf.org	| Extending IETF meetings to two weeks?	| The IETF meetings tend to become too large, cr...	| 1 |
|1	| 007-7484738 |	Wed, 09 Dec 1998 20:21:11 -0800	| Terry Allen	| discuss@apps.ietf.org,jpalme@dsv.su.se | Re: Extending IETF meetings to two weeks? |	> The IETF meetings tend to become too large, ...	| 2 |

