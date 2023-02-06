The purpose of this code is to be able to train gpt3 (accessed here through openai.Completion.create in python) on a 
large database. 
Note that gpt3 responds best to a "few shots" training. however, it is not possible to include the entire training 
data in the prompt (partly because the strongest engine (da-vinci) has a limit prompt+answer < ~4000 tokens). 
The solution is to:
-- create ahead of time the training database, from a .xlsx file, which also contains the embeddings of all prompts. 
This training database is stored in a pickle .pkl file (which has the advantage of preserving all data types). 
-- for a particular user's query, find top (n=5) similar prompts to that query (using cosine similarity of embeddings) 
and create a "training prompt" for GPT3 formed from these top (n=5) similar prompts and the query. Then pass this curated
query to GPT3 to obtain a completion. 

NOTE: This code can be improved / extended in (at least) 3 directions: 

1. Improve the training (.csv) database for the particular knowledge desired / tasks. 
2. Possibly find other ways (less expensive) for creating embeddings (openai charges 
