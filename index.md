
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

The structure of the XML file is like this:

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


Now we will create a function to extract information that we need and load it into a Pandas dataframe. We call this function `parse_bc3_emails`. It takes the XML content inside root loaded using ElementTree and returns a datafranme. The senetences are appended together to form the complete email text as a paragraph. Along with it, email metadata, *Recieved*, *From*, *To*, *Subject*, *Text* are also extracted. 

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
                for email_attribute in thread_element:
                    #If the email_attri is text, then each child contains a line from the body of the email
                    if email_attribute.tag == "Text":
                        email_body = ""
                        for sentence in email_attribute:
                            email_body += sentence.text
                    else:
                        #The attributes of the Email <Recieved, From, To, Subject, Text> appends in this order. 
                        email_metadata.append(email_attribute.text)
                        
                #Use same enron cleaning methods on the body of the email
                split_body = clean_body(email_body)
                    
                email_dict = {
                    "listno" : listno,
                    "date" : process_date(email_metadata[0]),
                    "from" : email_metadata[1].split('<')[0],
                    "to" : email_metadata[2],
                    "subject" : email_metadata[3],
                    "body" : split_body['body'],
                    "email_num": email_num
                }
                
                BC3_email_list.append(email_dict)           
    return pd.DataFrame(BC3_email_list)

```
The dataframe contains following columns:

- listno (used to link the summaries to the correct emails)
- date 
- from
- to
- subject
- body (text of the email)
- email_num (incremental count of emails)

This function uses other two helper fucntions inside it: `clean_body` and `process_date`:

*Function: clean_body*

```
def clean_body(mail_body):
    '''
    This extracts both the email signature, and the forwarding email chain if it exists. 
    '''
    delimiters = ["-----Original Message-----","To:","From"]
    
    #Trying to split string by biggest delimiter. 
    old_len = sys.maxsize
    
    for delimiter in delimiters:
        split_body = mail_body.split(delimiter,1)
        new_len = len(split_body[0])
        if new_len <= old_len:
            old_len = new_len
            final_split = split_body
            
    #Then pull chain message
    if (len(final_split) == 1):
        mail_chain = None
    else:
        mail_chain = final_split[1] 
    
    #The following uses Talon to try to get a clean body, and seperate out the rest of the email. 
    clean_body, sig = extract_signature(final_split[0])
    
    return {'body': clean_body, 'chain' : mail_chain, 'signature': sig}

```

