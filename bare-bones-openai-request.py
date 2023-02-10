import openai
import tiktoken
import pandas as pd
#from api import GPT

engine='text-davinci-003'
temperature=0.
top_p=1.
presence_penalty=0.
max_tokens=300

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

#acprompt = '''Using the blueprint tools, write steps to create a movie rentals api, 
#with collections of users,
#movies and users balance and with relations between these.'''
df = pd.read_pickle('blueprint-primer-large-ada-embeddings-with-tokens.pkl')
task = '''Learn about blueprint: '''

#query = "What is blueprint?"
query = '''Using the blueprint tools, write steps to create a shoe e-shopping api, 
with collections of users, products, shopping events and users balance, 
and with relations #between these.'''

actual_prompt=task
tokens = num_tokens_from_string(task+query, encoding_name="p50k_base")
print('the number of tokens in the task + query is: ', tokens)

N=df.shape[0]

print('We have ', N, ' (prompt, completion) pairs in the knowledge database')

for i in range(N):
     actual_prompt += 'input: '+ df.iloc[i]['prompt'] + ' output: ' + df.iloc[i]['completion']+' '+ '\n'
     tokens += df.iloc[i]['prompt_and_completion_tokens']

#print('GPT is prompted with ', prompt)
print('len(prompt) is ', len(actual_prompt))
print('Nr of tokens in prompt is ', tokens)

response = openai.Completion.create(engine=engine, prompt= actual_prompt, max_tokens= max_tokens,
           temperature=temperature, top_p=top_p, presence_penalty=presence_penalty, n=1)


print('The first 200 characters of the actual prompt are: \n')
print(actual_prompt[:200])
print('The last 100 characters of the actual prompt are: \n')
print(actual_prompt[-100:])
#print('GPT 3 engine is prompted with ', acprompt)
print('The GPT 3 engine responds with: ', response['choices'][0]['text']) 