The purpose of this code is to be able to train gpt3 (accessed here through openai.Completion.create in python) on a 
large database. 
Note that gpt3 responds best to a "few shots" training. however, it is not possible to include the entire training 
data in the prompt (partly because the strongest engine (da-vinci) has a limit prompt+answer < ~4000 tokens). 
The solution is to:
a) create ahead of time the training database, from a .xlsx file, which also contains the embeddings of all prompts. 
This training database is stored in a pickle .pkl file (which has the advantage of preserving all data types). 
b) for a particular user's query, find top (n=5) similar prompts to that query (using cosine similarity of embeddings) 
and create a "training prompt" for GPT3 formed from these top (n=5) similar prompts and the query. Then pass this curated
query to GPT3 to obtain a completion. 



The main working files are: 
1. embed_examples_with_tokens.py (for embedding the prompts in the base .xlsx training file)
and for calculating the (prompt, completion) tokens 
2. access_gpt3_for_large_few_shots_base.py for accepting a query, creating the prompt to be fed to the engine 
(by pre-pending the query with top 5 most similar prompts (and their completions)), feeding the created prompt
to the GPT3 engine and outputing the engine's completion. 

NOTE: This code can be improved / extended in one - two directions: 

1. Improve the training (.csv) database for the particular knowledge desired / tasks. 
2. Possibly find other ways (less expensive) for creating embeddings (openai charges for each created embedding, the charge is 0.2$ for 1K embedded token, as the most expensive charge, ada is much cheaper but I found small (accuracy, possibly floating point approximation) errors with ada's embeddings.



