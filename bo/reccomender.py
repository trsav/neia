import openai
import json
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import * 

with open("misc/api_key.txt") as f:
    openai.api_key = f.readline()


def expert_reccomendation(x_names,x,u,data,subject,objective_description,model,temperature):

    context =  " You are an expert in " + subject + "."
    context += " You are tasked with selecting the best solution from a set of " + str(len(x)) + " alternative solutions to achieve the goal of " + objective_description + "."
    context += " A set of alternative solutions to achieve this goal is provided to you."
    context += " What follows is a description of what is provided to you for each alternate solution: \n\n"

    context += " Decision variables (x): in order, these describe " + ''.join([x_names[i]+', ' for i in range(len(x_names)-1)])+ "and " + x_names[-1] + ".\n"
    context += " Utility (U(x)): the value of the acquisition function/utility function for a given solution. This value is calculated as a function of the predictive distribution of the objective of a solution.\n"
    context += " U(x) considers the exploration-exploration trade-off, where a higher value is more attractive, and theoretically a better choice.\n"
    context += " However, you must condition these value with your own expertise in " + subject + " which will inform your final decision."
    context += " As a large-language model you have access to additional information, physical insight, and real-world knowledge, that the calculation of utility did not consider."
    context += " Importantly, you must consider how each solution will perform in the real world and how it relates to the objective. The utility quantities provided have been calculated with no account of physical knowledge, you must be sceptical with respect to these values in light of your knowledge."
    context += " You must consider the relative differences between the information provided for each solution, and how this relates to the objective, as well as the physical differences between the solutions."
    context += " You must be completely neutral as to whether the physical knowledge you understand regarding the solutions outweighs the utility values, or vice versa."
    context += " You must think clearly, logically, and step-by-step to select the best option from the alternative solutions provided, selecting that one that you think will optimsation objective."


    user_prompt = f'''
    Variables (x): {''.join([x_names[i]+', ' for i in range(len(x_names)-1)])+ "and " + x_names[-1]}\n
    '''
    # rounding solutions to save tokens
    round = 3
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = np.round(x[i][j],round)

    for i in range(len(x)):
        sol_str = ''.join([x_names[j]+': '+ str(x[i][j]) +', ' for j in range(len(x[i]))])
        user_prompt += f'''Solution {str(i+1)}: {sol_str}, Utilty value, U(x) = {u[i]} \n'''
    user_prompt += '\nNote that higher values of U(x) are more attractive, and theoretically better choices.\n'
    objective = '\nOptimisation Objective: '  + objective_description
    user_prompt += objective + '\n'

    prev_data_len = len(data['previous_iterations'])

    user_prompt += f'''
    Below is a JSON object containing the previous {str(prev_data_len)} iterations of the optimisation process, the inputs are respective to the variables described above, and the outputs are the objective function values.
    This may include your previous justifiction given for selecting a datapoint. Your previous reasoning may or may not be correct, but it is important to consider the reasoning you gave for your previous selections.
    '''
    # round every value in data to save tokens
    clean_data = []
    for i in range(prev_data_len):
        x_clean = {}
        for j in range(len(data['previous_iterations'][i]['inputs'])):
            x_clean[x_names[j]] = np.round(data['previous_iterations'][i]['inputs'][j],round)

        clean_data.append({'inputs':x_clean,'objective':np.round(data['previous_iterations'][i]['objective'],round)})
        try:
            clean_data[i]['reason'] = data['previous_iterations'][i]['reason']
        except:
            continue
    data = {'previous_iterations':clean_data}
    user_prompt += json.dumps(data) + '\n'

    user_prompt += '''
    Provide your response as a JSON object ONLY. Do not include any additional text.
    The JSON object must contain the key "choice" and the value as the index of the best alternative. 
    The other key is named 'reason' and must be a brief and concise explanation of your reasoning (approx 50 words), with respect to the additional physical knowledge you have considered.
    '''
    print(user_prompt)
    messages=[
        {"role": "system", "content": context},
        {"role": "user", "content": user_prompt},
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

    response_message = response["choices"][0]["message"]['content']

    return response_message

# todo integrate with bo, and add alternatives to res. (will need to process this to not provide these to reccomender)
# todo so add "alternatives" and "reason" to data

# x_names = ["Temperature", "Pressure", "Catalyst Concentration"]
# expertise = "Reaction engineering & chemistry"
# obj_desc =  'Maximise the yield of B within a chemical reaction where A -> B -> C. The first reaction is exothermic, and the second is endothermic.'


# alternatives = 3
# round = 3 # round to save tokens
# x = [list(np.round(np.random.uniform(0,1,3),round)) for i in range(alternatives)]
# u = list(np.round(np.random.uniform(0,1,3),round))


# data_full = read_json('misc/synthetic_data.json')
# previous_iterations = 5
# temperature = 0.1
# model = 'gpt-3.5-turbo-0613'
# # data is the last previous iterations 
# data = {'previous_iterations':data_full['data'][-previous_iterations:]}
# response = json.loads(expert_reccomendation(x_names,x,u,data,expertise,obj_desc,model,temperature))
# print(response)
# print(response['choice'])
# print(response['reason'])