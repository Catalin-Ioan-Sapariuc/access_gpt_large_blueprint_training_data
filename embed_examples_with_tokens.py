## read (prompt, completion) pairs from csv file . 
## embed the prompt using the text-similarity-davinci-001 engine embedding (openai)
## then saved the full embeedded df in a new .csv file (blueprint-primer-general-and-specific-embeddings.csv)

import pickle
import openai
import tiktoken
import pandas as pd
import numpy as np

openai.api_key = "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"
#embedding_engine = "text-similarity-davinci-001"
embedding_engine = "text-embedding-ada-002" 
# possible engines (encoding_names) to consider for num_tokens_from_string
# are: gpt2 (for all gpt3 embeddings), "p50k_base" (for text-davinci-002 and text-davinci-003 and
# "cl100k_base" for text-embedding-ada-002 

def text_embed(text: str) -> str:
    response = openai.Embedding.create(input=text,engine=embedding_engine)
    curated_response = response.data[0]["embedding"]
    return curated_response

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

df = pd.read_excel('blueprint-primer-large.xlsx')

#print('the size of original df from blueprint-primer-large.xlsx \n')
print(df.shape)

df.replace(r'\s+|\\n', ' ', regex=True, inplace=True)
#df.replace(r'\s+|\"', '', regex=True, inplace=True)

#print('the df head after replacing all the newlines with spaces \n')
print("the df head before adding the prompt and completion tokens \n")
print(df.head(3))

df['prompt_embedding']=df['prompt'].map(text_embed)

df['prompt_and_completion_tokens']=df.apply(lambda x: num_tokens_from_string(x['prompt'], "p50k_base" ) + num_tokens_from_string(x['completion'],"p50k_base"), axis=1)

print('the df head after adding the prompt and completion tokens \n')
print(df.head(5))

df.to_pickle('blueprint-primer-larger-ada-embeddings-with-tokens.pkl')

