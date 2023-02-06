## read (prompt, completion) pairs from csv file . 
## embed the prompt using the text-similarity-davinci-001 engine embedding (openai)
## then saved the full embeedded df in a new .csv file (blueprint-primer-general-and-specific-embeddings.csv)

import pickle
import openai
import pandas as pd
import numpy as np

embedding_engine = "text-similarity-davinci-001"
#embedding_engine = "text-embedding-ada-002" 

def text_embed(text):
    response = openai.Embedding.create(input=text,engine=embedding_engine)
    curated_response = response.data[0]["embedding"]
    return curated_response

df = pd.read_excel('blueprint-primer-general-and-specific.xlsx')

print('the size of original df from blueprint-primer-general-and-specific.csv \n')
print(df.shape)

#df.replace(r'\s+|\\n', ' ', regex=True, inplace=True)
#df.replace(r'\s+|\"', '', regex=True, inplace=True)

#print('the df head after replacing all the newlines with spaces \n')
#print(df.head(3))

df['prompt_embedding']=df['prompt'].map(text_embed)


df.to_pickle('blueprint-primer-general-and-specific-embeddings.pkl')

