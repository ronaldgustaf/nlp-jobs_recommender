# Job Recommender System using TF-IDF Vectorizer and Word2Vec
Training Dataset: https://www.kaggle.com/datasets/promptcloud/indeed-usa-job-listing-dataset
> Filename: marketing_sample_for_indeed_usa-indeed_usa_job__20210101_20210331__30k_data.ldjson

Sample Resume Dataset: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
filtered for categories: ['Data Science', 'Web Designing', 'HR', 'Business Analyst', 'Civil Engineer']
> Filename: resume_data.csv

## NLP Text Preprocessing
- Spacy
- Regex

## Splitting Data
80% train, 20% test

## Creating Recommender System using TF-IDF Vectorizer and self-trained Word2Vec
Performance metric: Normalized Discounted Cumulative Gain (NDCG)
![ndcg](/images/ndcg.png "ndcg")
> Reference: https://towardsdatascience.com/evaluate-your-recommendation-engine-using-ndcg-759a851452d1

### TF-IDF Vectorizer
- Fit and transform train data preprocessed sentences to sklearn's TF-IDF Vectorizer
- Transform test data processed sentences
- Compute cosine similarity between test and train set to get the similarity matrix
- Calculate NDCG Score of Top 50 similar jobs recommended
TF-IDF Mean NDCG: 0.9898483787830057

### Word2Vec
- Train Word2Vec from the train data preprocessed sentences of job descriptions
> 'job_embeddings_word2vec_200_w10.model'
- Tokenize text using Keras' Tokenizer
- Pad the sequences with zeros
- Convert matrix to scipy's csr sparse matrix
- Create word embedding matrix
- Prepare test data sequences
- Compute cosine similarity between test and train set to get the similarity matrix
- Calculate NDCG Score of Top 50 similar jobs recommended
Word2Vec Mean NDCG: 0.9859520623803626

## Testing Recommender System on Resume Content using TF-IDF Vectorizer
> Dataset reference: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset?resource=download
- Retrieve 1 resume each for 5 categories
> categories = ['Business Analyst', 'Civil Engineer', 'Data Science', 'HR', 'Web Designing']
- Find Top 10 Jobs according to CV content
- Results:

    Business Analyst Resume:
    ![ba](/images/ba.png "ba")

    Civil Engineer Resume:
    ![civ](/images/civ.png "civ")

    Data Scientist Resume:
    ![ds](/images/ds.png "ds")

    HR Resume:
    ![hr](/images/hr.png "hr")

    Web Designing Resume:
    ![webdev](/images/webdev.png "webdev")

> Notes: Content of Web Designing Resume tested
    ['Technical Skills Web Technologies: Angular JS, HTML5, CSS3, SASS, Bootstrap, Jquery, Javascript. Software: Brackets, Visual Studio, Photoshop, Visual Studio Code Education Details ',
    'January 2015 B.E CSE Nagpur, Maharashtra G.H.Raisoni College of Engineering',
    'October 2009  Photography Competition Click Nagpur, Maharashtra Maharashtra State Board',
    'College Magazine OCEAN',
    'Web Designer ',
    'Web Designer - Trust Systems and Software',
    'Skill Details ',
    'PHOTOSHOP- Exprience - 28 months',
    'BOOTSTRAP- Exprience - 6 months',
    'HTML5- Exprience - 6 months',
    'JAVASCRIPT- Exprience - 6 months',
    'CSS3- Exprience - Less than 1 year months',
    'Angular 4- Exprience - Less than 1 year monthsCompany Details ',
    'company - Trust Systems and Software',
    'description - Projects worked on:',
    '1. TrustBank-CBS',
    'Project Description: TrustBank-CBS is a core banking solution by Trust Systems.',
    'Roles and Responsibility:',
    'â\x97\x8f Renovated complete UI to make it more modern, user-friendly, maintainable and optimised for bank use.',
    'â\x97\x8f Shared the UI structure and guidelines to be incorporated, with development team of around 50',
    'members.',
    'â\x97\x8f Achieved the target of project completion in given time frame.',
    'â\x97\x8f Made required graphics for the project in photoshop',
    '',
    '2. Loan Bazar (Loan Appraisal)',
    'Project Description: Loan Bazar is a MVC-based application dedicated to creating and managing',
    'loan applications. The goal of this application is to streamline the process of loan application and integrate with existing CBS.',
    'Roles and Responsibility',
    'â\x97\x8f Designed and developed modern and responsive UI of entire application and achieved the target in given time frame.',
    'â\x97\x8f Made required graphics for the project in photoshop',
    '3. Capital Security Bond Application',
    'Project Description: Capital Security Bond Application is a MVC based application which provided an online platform to purchase gold bond',
    'Roles and Responsibility:',
    'â\x97\x8f Designed and developed modern and responsive UI of entire application and achieved the target in given time frame.',
    'â\x97\x8f Made required graphics for the project in photoshop',
    '',
    '4. SoftGST',
    'Project Description: SoftGST (Web Based Application) is an ASP application to every tax',
    'payers and its vendors for generating the GSTR returns on the basis of sales / purchase',
    'data, additionally the application can do the reconciliation of GSTR 2 A with purchase register.',
    'Roles and Responsibility:',
    'â\x97\x8f Designed and developed the UI of Dashboard.',
    '',
    '5. Trust Analytica:',
    'Project Description: Trust Analytika is the mobile web app that shows bank asset, liability,',
    'income, expenses.',
    'Roles and Responsibility:',
    'â\x97\x8f Designed and developed the landing page of the application.',
    'â\x97\x8f Supported the developers in UI implementation',
    '',
    "6. Website's:",
    'Project Name:',
    '1. TSR Technology Services - http://tsrtechnologyservices.com',
    '2. Vidarbha Merchants Urban Co-Op Bank - http://vmcbank.com',
    '3. GISSS - http://gisss.co.in',
    '4. Softtrust USA - http://softtrustusa.com',
    'Roles and Responsibility',
    'â\x97\x8f Communicated with clients to understand their requirement',
    'â\x97\x8f Made mocks for the website',
    'â\x97\x8f Designed and developed complete website and hosted them in stipulated time.',
    'company - www.jalloshband.com',
    'description - Project Name:',
    '1. Jallosh Band - www.jalloshband.com',
    '2. An Endeavor Foundation',
    'Roles and Responsibility:',
    'â\x97\x8f Communicated with clients to understand their requirement',
    'â\x97\x8f Made mocks for the website',
    'â\x97\x8f Designed and developed complete website and hosted them in stipulated time.',
    'company - 10MagicalFingers',
    'description - National and international client interaction.',
    'â\x97\x8f Management of digital data']
