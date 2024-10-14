import openai
import pandas as pd
import openai
import json
import numpy as np
# set API-key
openai.api_key = ''



# calculate similarity
def cosine_similarity(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos_sim


# calculate MSE
def mean_squared_error(v1, v2):
    mse = np.mean((v1 - v2) ** 2)
    return mse


# ask a question
question = "Al: 3% , Zn: 42%, temperature: 695℃"
try:
    # get embedding vector of question
    response = openai.Embedding.create(
            model="text-embedding-ada-002",  # 可以选择合适的嵌入模型
            input=question
        )
        # extract embedding vector
    embeddings = response['data'][0]['embedding']
except Exception as e:
    print(e)
    embeddings = None
    raise Exception('Failed to get embeddings')
# create a matrix
matrix = np.load('D:/vscode/PDGPT/all_embeddings.npy')
phrases = open('D:/vscode/PDGPT/all_embeddings.txt', encoding='utf-8').readlines()

# similarities = mean_squared_error(matrix, embeddings)
similarities = np.array([cosine_similarity(row, embeddings) for row in matrix])
mses = np.array([mean_squared_error(row, embeddings) for row in matrix])

# Top k
top_indices_sim = np.argsort(similarities)[::-1][:2]
print("cos_sim")
print(top_indices_sim)
top_coordinates = []
for i in top_indices_sim:
    print(phrases[i])
    print(similarities[i])
    for j in range(i,i+10):
        top_coordinates.append(phrases[j])

print(top_coordinates)
# print("```````````````````````")
# print('mse')
# top_indices_mse = np.argsort(mses)[:10]

# print(top_indices_mse)
# top_coordinates = []
# for i in top_indices_mse:
#     print(phrases[i])
#     top_coordinates.append(phrases[i])


# print(top_coordinates)

# Answer the question
response = openai.ChatCompletion.create(
    model="gpt-4o",  # choose model version
    messages=[
        {"role": "system", "content": "You are an expert in phase diagrams and thermodynamics, specializing in the phase equilibria of material systems (such as binary and ternary phase diagrams). You have a deep understanding of eutectic reactions, peritectic reactions, and liquid-solid phase transformations, and you can calculate and predict phase stability under various conditions. Your goal is to provide accurate, clear, and correct answers to questions related to phase transformations, solid solutions, and multi-phase equilibria, prioritizing correctness, while only showing the key steps of the reasoning process. When users provide you with the composition and temperature conditions, you can accurately determine the phase composition under those conditions. Whether the user is seeking phase diagram analysis, thermodynamic background knowledge, or material optimization advice, you can give precise answers. Note: Even if you are uncertain, you must provide a clear answer to help the user evaluate the accuracy. Responses should be output in JSON format."},
        {"role": "user", "content": f"Based on the information of top coordinates: {top_coordinates}, please answer the question: {question}, to find an answer that is close to the question. You must ensure that the Al and Zn content, as well as the temperature, match the conditions specified in the question. If there is no exact match in the knowledge base, you must infer a clear answer based on the most relevant knowledge. Leaving the question unanswered is not allowed."},
    ]
)

# print the answer
print(response['choices'][0]['message']['content'])