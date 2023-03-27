import openai
import tiktoken
import pandas as pd
#from api import GPT

model ='gpt-3.5-turbo'
#temperature=0.
#top_p=1.
#presence_penalty=0.
#max_tokens=300

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

#acprompt = '''Using the blueprint tools, write steps to create a movie rentals api, 
#with collections of users,
#movies and users balance and with relations between these.'''
df = pd.read_pickle('new-blueprint-primer-embeddings-with-tokens-mac.pkl')
#task = '''Learn about blueprint: '''

query = "What is blueprint?"
#query = '''Using the blueprint tools, write steps to create a shoe e-shopping api, 
#with collections of users, products, shopping events and users balance, 
#and with relations #between these.'''

actual_prompt = []
tokens = num_tokens_from_string(query, encoding_name="p50k_base")
print('the number of tokens in the query is: ', tokens)

N=df.shape[0]

print('We have ', N, ' (prompt, completion) pairs in the knowledge database')

for i in range(N):
     actual_prompt.append({"role": "user", "content": df.iloc[i]['prompt']})
     actual_prompt.append({"role" : "assistant", "content": df.iloc[i]['completion']})

actual_prompt.append({"role": "user", "content": query})

#print('GPT is prompted with ', prompt)
print('len(prompt) is ',actual_prompt)
#print('Nr of tokens in prompt is ', tokens)

response = openai.ChatCompletion.create(model = model, messages = actual_prompt)

print('The first 200 characters of the actual prompt are: \n')
print(actual_prompt[:200])
print('The last 100 characters of the actual prompt are: \n')
print(actual_prompt[-100:])
#print('GPT 3 engine is prompted with ', acprompt)
print('The GPT 3 engine responds with: ', response['choices'][0]['text']) 