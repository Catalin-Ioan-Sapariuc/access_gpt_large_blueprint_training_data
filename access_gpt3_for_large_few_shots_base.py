''' 
Having a knowledge database of prompts, completions and embeddings of prompts.
ask user for query (or introdce a query), then choose the top (n=5) of most 
similar prompts (with the query) from the knowledge database, 
create  new acc_prompt made of the similar (prompt, completion) pairs and the query. 
Finally, feed this acc_prompt to the GPT-3 engine to generate the completion. 

Ioan Sapariuc
Feb 2023
''' 
import tiktoken
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
frequency_penalty=0.
max_tokens=300

def text_embed(text:str) -> str:
    response = openai.Embedding.create(input=text,engine=embedding_engine)
    curated_response = response.data[0]["embedding"]
    return curated_response

def cosine_similarity(A,B):
    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

df = pd.read_pickle('blueprint-primer-large-ada-embeddings-with-tokens.pkl')

#print(df.head(3))
#print(type(df['prompt_embedding'][0]))
task = '''Learn about blueprint: '''

query = "What is blueprint?"
#query = '''Using the blueprint tools, write steps to create a shoe e-shopping api, 
#with collections of users, products, shopping events and users balance, 
#and with relations between these.'''

#query = ''' Using the blueprint tools, write .ts code to create collections of users and products 
#for an e-shopping api, and with relations between these.'''

query_embed = text_embed(query)

df['query_similarity']=df.apply(lambda x: cosine_similarity(x['prompt_embedding'], query_embed), axis=1)
#print('similarity between query and the first prompt in the knowledge database is: ', df['query_similarity'][0])


#print('the df head after adding the query similarity \n')
#print(df.head(3))

#dfs = df.sort_values(by='query_similarity',ascending=False, ignore_index=True)
dfs = df.sort_values(by='query_similarity',ascending=False)
#n=7  ## how many similar prompts to use (from the knowledge database)
print('the df(10) head after sorting by query similarity \n')
print(dfs.head(10))

actual_prompt=task
tokens = num_tokens_from_string(task+query, encoding_name="p50k_base")
print('the number of tokens in the task + query is: ', tokens)

n=0

while True: 
    if (tokens +dfs.iloc[n]['prompt_and_completion_tokens'] < 4000 - max_tokens):
        actual_prompt += 'input: '+ dfs.iloc[n]['prompt'] + ' output: ' + dfs.iloc[n]['completion']+' '+ '\n'
        tokens += dfs.iloc[n]['prompt_and_completion_tokens']
        n +=1
    else:
        break

print('we added ', n, ' (prompts, completions), where prompts are the most similar to the actual prompt')
print('the number of tokens in the actual prompt is: ', tokens)

actual_prompt += 'input: ' + query + ' output: '

#print(len(actual_prompt))
if (len(actual_prompt) > 10000):
    print('warning: you are reaching the max allowed number of tokens since len(actual_prompt)= ', len(actual_prompt))

print('The first 200 characters of the actual prompt are: \n')
print(actual_prompt[:200])
print('The last 100 characters of the actual prompt are: \n')
print(actual_prompt[-100:])
#print('the actual prompt is: \n ')
#print(actual_prompt)

response = openai.Completion.create(model=engine, prompt= actual_prompt, max_tokens= max_tokens,temperature=temperature, presence_penalty=presence_penalty, 
frequency_penalty=frequency_penalty, stop=["input:"])


#response = openai.Completion.create(
#  model="text-davinci-003",
#  prompt=actual_prompt,
#  temperature=0,
##  max_tokens=max_tokens,
#  top_p=1,
#  frequency_penalty=0,
##  presence_penalty=0,
 # stop=["input:"]
#)

print('The GPT 3 engine responds with: ', response['choices'][0]['text']) 




