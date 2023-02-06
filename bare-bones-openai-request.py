import openai

#from api import GPT

engine='text-davinci-003'
temperature=0.
top_p=1.
presence_penalty=0.
max_tokens=300


#acprompt = '''Using the blueprint tools, write steps to create a movie rentals api, 
#with collections of users,
#movies and users balance and with relations between these.'''
acprompt="input: What is blueprint?, output: "

ex1= '''input: Description of blueprint, output: Blueprint is a paradigm that lets you write schema code and it can
generate code while at the same time giving you the ability to add features, modify code and 
also support additional functionalities. Because Blueprint outputs an X-Framework compatible project, it's important
that you have you are up to date with the X-Framework basics. \n''' 

ex2= '''input: Pre-requisites for building a project in blueprint, output:
     1. Have node.js 14+ and mongodb installed \n 2. Have mongodb service running \n '''; 

ex3= '''input: Create a new project, output: 
1. npm i -g i -g bluelibs/x (installs the x framework) \n 
2. x then choose x:project (creates a project using x) \n 3. cd project npm install\n ''' 


ex4 = '''input: Create the basic structure of the project, output:
npm run blueprint:generate \n 
This creates two microservices: api, which contains
MongoDB and graphQL interfaces powered by XFramework and the admin powered by Webpack, 
React and Ant Design \n 
'''

ex5 = '''input: Customize the api:, output:
1. under blueprint/collections :
edit User.ts as needed (add fields, relations, behaviors etc) \n
2. create a collection in blueprint/collections and add fields, for example:
collection({ id: "Tasks", description: "A collection for TO DO list items", fields:
[ // Here we specify the fields for collection field({ id: "title", type: fields.types.STRING, }), 
field({ id: "description", type: fields.types.STRING, }),
 field({ id: "softDeadline", type: fields.types.DATE, }), 
 field({ id: "hardDeadline", type: fields.types.DATE, }),
 field({ id: "createdAt", type: fields.types.DATE, defaultValue: new Date(), }), ], }) \n'''

ex6= '''input: Generate the customized project:, output: npm run blueprint:generate \n '''

ex7 = '''input: Add behaviors or relations to the collection:", output: fill collection({ id: "Tasks", relations[]) \n '''

prompt = ex1 + ex2 + ex3 + ex4 + ex5 + ex6 + ex7 + acprompt 

#print('GPT is prompted with ', prompt)
print('len(prompt) is ', len(prompt))

response = openai.Completion.create(engine=engine, prompt= prompt, max_tokens= max_tokens,
           temperature=temperature, top_p=top_p, presence_penalty=presence_penalty, n=1)

print('GPT 3 engine is prompted with ', acprompt)
print('The GPT 3 engine responds with: ', response['choices'][0]['text']) 