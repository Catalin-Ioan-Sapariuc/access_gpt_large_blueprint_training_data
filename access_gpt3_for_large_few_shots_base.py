''' 
Having a knowledge database of prompts, completions and embeddings of prompts.
ask user for query (or introdce a query), then choose the top (n=5) of most 
similar prompts (with the query) from the knowledge database, 
create  new acc_prompt made of the similar (prompt, completion) pairs and the query. 
Finally, feed this acc_prompt to the GPT-3 engine to generate the completion. 

Ioan Sapariuc
Feb 2023
''' 

import pickle
import openai
import pandas as pd
import numpy as np

#embedding_engine = "text-similarity-davinci-001"
embedding_engine = "text-embedding-ada-002" 
engine='text-davinci-003'
temperature=0.
top_p=1.
presence_penalty=0.
max_tokens=600

def text_embed(text:str) -> str:
    response = openai.Embedding.create(input=text,engine=embedding_engine)
    curated_response = response.data[0]["embedding"]
    return curated_response

def cosine_similarity(A,B):
    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

df = pd.read_pickle('blueprint-primer-general-and-specific-embeddings-with-tokens.pkl')

#print(df.head(3))
#print(type(df['prompt_embedding'][0]))

#query = "What is blueprint?"
query = '''Using the blueprint tools, write steps to create an e-shopping ai, 
with collections of users,
products, shopping events and users balance, and with relations between these.'''

query_embed = text_embed(query)

df['query_similarity']=df.apply(lambda x: cosine_similarity(x['prompt_embedding'], query_embed), axis=1)
#print('similarity between query and the first prompt in the knowledge database is: ', df['query_similarity'][0])


#print('the df head after adding the query similarity \n')
#print(df.head(3))

dfs = df.sort_values(by='query_similarity',ascending=False)

#print('the df head after sorting by query similarity \n')
#print(dfs.head(4))

n=5; ## how many similar prompts to use (from the knowledge database)
actual_prompt=''
for i in range(n):
    actual_prompt += 'input: '+ dfs['prompt'][i]+ ' output: ' + dfs['completion'][i]+' '+ '\n'
actual_prompt += 'input: ' + query + ' output: '

#print(len(actual_prompt))
if (len(actual_prompt) > 10000):
    print('warning: you are reaching the max allowed number of tokens since len(actual_prompt)= ', len(actual_prompt))

#print('The first 200 characters of the actual prompt are: \n')
#print(actual_prompt[:200])
#print('The last 100 characters of the actual prompt are: \n')
#print(actual_prompt[-100:])

response = openai.Completion.create(engine=engine, prompt= actual_prompt, max_tokens= max_tokens,
           temperature=temperature, top_p=top_p, presence_penalty=presence_penalty, n=1, stop="\n\ninput:")


print('The GPT 3 engine responds with: ', response['choices'][0]['text']) 




