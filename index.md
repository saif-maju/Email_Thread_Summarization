
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
Humans can easily summarize any text, such as an essay or article, by just reading it, noting or highlighting key points and then parapharsing it in their own words. This seems pretty simple task to do. However, machine cannot learn so easily. NLP Researchers have tried different methods over time to make it learn how to summrize any text. Two of the major approaches still used are described below.  

## Extractive Summarization
With this approach, we use the first step from a human course of actions: Highlighting the important points. In this technique, we select most informative senetences from the text and use them as a summary. This is a naive approach but in most cases it produces very good results,  as often some sentences provide as much information as necessary to understand the whole text.

We have applied this technique using TextRank Alogorithm which we will cover in the Modeling section.

## Abstractive Summarization
This approach is more humanly as it not only remembers key points as well as reproduces them in natural language. It is a difficult task and requires complex language modeling. This is why we use state-of-the-art T5 model to model this. 
This will be the focus of our article. 

# 2. Dataset Description

## Colossal Clean Crawled Corpus (C4)

This dataset is a shortened and cleaned version of [Common Crawl Dataset](https://commoncrawl.org/). It was reduced from 20TB to just 75GB by applying some heuristics to clean the text. It is web extracted text which contained content in HTML pages, with markup and non-text content removed. Google as used data from April 2019 for pre-training T5 and released it as part of [TensorFlow Datasets](https://www.tensorflow.org/datasets/).

## British Columbia Conversation Corpora (BC3)
This [dataset](https://www.cs.ubc.ca/cs-research/lci/research-groups/natural-language-processing/bc3.html) is a collection of 40 email threads (3222 sentences) from W3C corpus. This is prepared for summarization task, annotated by three different annotators. They provided both extractive and abstractive summaries which could be used for training the model as a target summary. A number of researchers have used this dataset for serveral NLP tasks.

We will use this dataset for fine-tuning T5. Another section of this article will discuss preprocessing of this dataset. 

# 4. Data Extraction and Preprocessing

The dataset can be downloaded from website of [The University of British Columbia](https://www.cs.ubc.ca/cs-research/lci/research-groups/natural-language-processing/bc3.html). It is also available on our [GitHub Repo](https://github.com/saif-maju/Email_Thread_Summarization). Once downloaded, we can preprocess it in Python. 

This dataset is split into two XML files: `corpus.xml` and `annotation.xml`. First contains the original emails split line by line, and the other contains the summaries created by the annotators. Each email may contain several summaraies from different annotators and summaries may also be over several emails. 

The structure of the `corpus.xml` file is like this:

```
<root>
  <thread>

    <name>Extending IETF meetings to two weeks?</name>
    <listno>007-7484738</listno>
    <DOC>
      <Received>Tue Dec 08 07:30:52 -0800 1998</Received>
      <From>Jacob Palme &lt;jpalme@dsv.su.se&gt;</From>
      <To>discuss@apps.ietf.org</To>
      <Subject>Extending IETF meetings to two weeks?</Subject>
      <Text>
        <Sent id="1.1">The IETF meetings tend to...</Sent>
        <Sent id="1.2">I suggest that future meetings...</Sent>
      </Text>
    </DOC>

    <DOC>
      <Received></Received>
      <From></From>
      <To></To>
      <Subject></Subject>
      <Text>
        <Sent id="2.1"></Sent>
         ...
      </Text>
    </DOC>

  </thread>
</root>
```
And, The structure of the `annotation.xml` file looks like this:


```
<root>
  <thread>
    <listno>067-11978590</listno>
    <name>AU WG joining us for an afternoon at F2F in Bristol?</name>
    <annotation>
      <desc>Annotator7-Part2</desc>
      <date>Fri May 23 14:01:53 -0700 2008</date>
      <summary>
        <sent link="1.1">Wendy states that they are moving forward with a face to face meeting in Bristol, England on Oct 5-6.</sent>
        <sent link="1.2">The Authoring tool group will join them to discuss revisions to the WCAG on the 6th.</sent>
        <sent link="3.1,3.2,3.3,3.5">Jonathan asks he is attending the venue, what the agenda is and if it would be useful if he attended with a presentation.</sent>
      </summary>
      <sentences>
        <sent id="1.1"/>
        <sent id="1.2"/>
        <sent id="1.3"/>
      </sentences>
      <labels>
        <prop id="1.2"/>
        ...
        <meet id="1.1"/>
        ...
        <req id="1.3"/>
        ..
        <subj id="2.3"/>
        ..
        <cmt id="7.6"/>
      </labels>
    </annotation>
```


Since emails are in XML format, we need to parse them. For this we will use ElementTree package, that has functions to read and manipulate XMLs. 

First, import ElementTree. It's a common practice to use the alias of ET:

```
import xml.etree.ElementTree as ET
```
Then you need to read in the file with ElementTree.

```
parsedXML = ET.parse( "/BC2_Email_Corpus/corpus.xml" )
root = parsedXML.getroot()
```


Now we will create a function to extract information that we need and load it into a Pandas dataframe. We call this function `parse_bc3_emails`. It takes the XML content inside root loaded using ElementTree and returns a datafranme. The senetences are appended together to form the complete email text as a paragraph. Along with it, email metadata, *Recieved*, *From*, *To*, and *Subject* are also extracted.  Before calling this function, we will import necessary packages. We will talk about it later in `clean_body` function.

```
# for storing and manipulating data
import pandas as pd
# for tokenization and selecting only valid words
import spacy
nlp = spacy.load('en_core_web_sm')
```

*Function: parse_bc3_emails*


```

def parse_bc3_emails(root):
    '''
    This adds every BC3 email to a newly created dataframe. 
    '''
    BC3_email_list = []
    #The emails are seperated by threads.
    for thread in root:
        email_num = 0
        #Iterate through the thread elements <name, listno, Doc>
        for thread_element in thread:
            #Getting the listno allows us to link the summaries to the correct emails
            if thread_element.tag == "listno":
                listno = thread_element.text
            #Each Doc element is a single email
            if thread_element.tag == "DOC":
                email_num += 1
                email_metadata = []
                sender = thread_element.find(".//From").text.lower()
                subject = thread_element.find(".//Subject").text.lower()
                email_tag = thread_element.find(".//Text")
                        
                subject_doc = nlp(subject)
                subject = ' '.join([token.text.lower() for token in subject_doc if token.is_alpha and not token.is_punct])

                #Use same enron cleaning methods on the body of the email
                cleaned_body = clean_body(email_tag,sender)

                email_dict = {
                    "listno" : listno,
                    "from" : sender.split()[0],
                    "subject" : subject,
                    "body" : cleaned_body,
                    "email_num": email_num
                }
                
                BC3_email_list.append(email_dict)           
    return pd.DataFrame(BC3_email_list)
```

The dataframe contains following columns:

- Listno: Thread identifier <br>
- From: The original sender of the email <br>
- Subject: Title of email. <br>
- Body: Original body. <br>
- Email_num: Email in thread sequence <br>


This function uses other two helper fucntions inside it: `clean_body` and `process_date`:

*Function: clean_body*

```
def clean_body(email_tag,sender):
  sender_list = sender.split()
  sender_list[-1] = sender_list[-1].replace('>','').replace('<','')
  email_stoppers = [l for l in sender_list if len(l)>2] + ['regards','thanks', 'çheers']
  clean_email = ''
  for email_item in email_tag:
    sent = nlp(email_item.text)
    clean_sent = ' '.join([token.text.lower() for token in sent if token.is_alpha and not token.is_punct])
    if sum([1 for word in email_stoppers if word in clean_sent and len(clean_sent.split())<=10])>=1:
        break 
    clean_email+=clean_sent+' '
  return clean_email

```
This function takes email tag element and sender (From tag) as input and returns cleaned body, with extra text, such as signature removed. For extracting signature from an email by limiting emails to the first occurence of any `email_stopper`. Here we use SpaCY package to apply following operations:

- tokenization
- checking if token alphabetical 
- checking if token is not a punctuation mark

Now we call the function we created.

```
bc3_email_df = parse_bc3_emails(root)
bc3_email_df.head(2)

```
*Output:* 

|listno |	from  |	subject	| body |	email_num |
| ---   | ---   | --- | --- | --- |
| 007-7484738 |	jacob| Extending IETF meetings to two weeks?	| the ietf meetings tend to become too large cre...		| 1 |
| 007-7484738 | Terry Allen | re extending ietf meetings to two weeks	 |	the ietf meetings tend to become too large cre...	| 2 |
| |



We define another similar function for parsing `annotations.xml`. THis function takes root as input and returns a dataframe containing:

- Annotator: Person who created summarization. <br>
- Email_num: Email in thread sequence. <br>
- Listno: Thread identifier. <br>
- Summary: Human summarization of the email. <br>

```
def parse_bc3_summaries(root):
    '''
    This parses every BC3 Human summary that is contained in the dataset. 
    '''
    BC3_summary_list = []
    for thread in root:
        #Iterate through the thread elements <listno, name, annotation>
        for thread_element in thread:
            if thread_element.tag == "listno":
                listno = thread_element.text
            #Each Doc element is a single email
            if thread_element.tag == "annotation":
                for annotation in thread_element:
                #If the email_attri is summary, then each child contains a summarization line
                    if annotation.tag == "summary":
                        summary_dict = {}
                        for summary in annotation:
                            #Generate the set of emails the summary sentence belongs to (often a single email)
                            email_nums = summary.attrib['link'].split(',')
                            s = set()
                            for num in email_nums:
                                s.add(num.split('.')[0].strip()) 
                            #Remove empty strings, since they summarize whole threads instead of emails. 
                            s = [x for x in set(s) if x]
                            for email_num in s:
                                if email_num in summary_dict:
                                    summary_dict[email_num] += ' ' + summary.text
                                else:
                                    summary_dict[email_num] = summary.text
                    #get annotator description
                    elif annotation.tag == "desc":
                        annotator = annotation.text
                #For each email summarizaiton create an entry
                for email_num, summary in summary_dict.items():
                    email_dict = {
                        "listno" : listno,
                        "annotator" : annotator,
                        "email_num" : email_num,
                        "summary" : summary
                    }      
                    BC3_summary_list.append(email_dict)
    return pd.DataFrame(BC3_summary_list)
```
Using this function, we parse summaries, load in a dataframe `bc3_summary_df` and merge it with `bc3_semail_df` using `email_num` and  `listno` columns. We call resulting dataframe `bc3_df`. 


```
#Load summaries and process
parsedXML = ET.parse( "/BC2_Email_Corpus/annotation.xml" )

root = parsedXML.getroot()

bc3_summary_df = parse_bc3_summaries(root)
bc3_summary_df['email_num'] = bc3_summary_df['email_num'].astype(int)

#merge the dataframes together
bc3_df = pd.merge(bc3_email_df, 
                  bc3_summary_df[['annotator', 'email_num', 'listno', 'summary']],
                  on=['email_num', 'listno'])
bc3_df.head(2)
```
*Output:* 

|listno	|	from  |	subject	| body |	email_num | annotator | summary |
| ---   | --- | --- | --- | --- | --- | --- | 
| 007-7484738 |	Jacob Palme	| Extending IETF meetings to two weeks?	| The IETF meetings tend to become too large, cr...	| 1 | Annotator3-Part2 | Jacob suggested to hold two week meetings, the... |
| 007-7484738 | Terry Allen	| Re: Extending IETF meetings to two weeks? |	> The IETF meetings tend to become too large, ...	| 2 | Annotator2-Part2 | Jacob suggested to hold two week meetings, the... |
|     |


We have got `annotator` and `summary` columns in `bc3_df` dataframe. Since we have two or three summaries for each email, we pick only first summary for each email. 

```
emails_df=bc3_df.groupby(['listno','email_num']).first()
emails_df['listno'] = emails_df.index.get_level_values(0)
emails_df['email_num'] = emails_df.index.get_level_values(1)
emails_df=emails_df.reset_index(drop=True)
emails_df.head()
```
*Output:*

|from|subject|body|annotator|summary|listno|email_num|
|---|---|---|---|---|---|---|
|jacob|extending ietf meetings to two weeks|the ietf meetings tend to become too large cre...|Annotator3-Part2|jacob suggested to hold two week meetings the ...|007-7484738|1|
|terry|re extending ietf meetings to two weeks|the ietf meetings tend to become too large cre...|Annotator3-Part2|jacob suggests that future ietf meetings be sp...|007-7484738|2|
|brian|re extending ietf meetings to two weeks|terry wg chairs already are asked to specify w...|Annotator3-Part2|the topic is the logistics of scheduling ietf ...|007-7484738|3|
|larry|create final ietf agenda schedule earlier|working groups do seem to decide at the last m...|Annotator3-Part2|terry supported jacob idea and suggested a fir...|007-7484738|4|
|richard|re create final ietf agenda schedule earlier|for example it would be very useful so that th...|Annotator3-Part2|some argue that it be more useful to prepare ...|007-7484738|5|
| |











