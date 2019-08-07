import json
import math
import random

def appendToFile(textFile, data):
    input = open(textFile, 'a+')
    input.write(data)

def openFile(textFile):
    input = open(textFile)
    return input

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def cost(prediction, target):
    return (prediction - target) ** 2

# ['gender'] => 0 = male | 1 = female
data = [
    {
        'height': 180,
        'age': 20,
        'gender': 0
    },
    {
        'height': 165,
        'age': 18,
        'gender': 1
    },
    {
        'height': 195,
        'age': 26,
        'gender': 0
    },
    {
        'height': 143,
        'age': 20,
        'gender': 1
    },
    {
        'height': 120,
        'age': 19,
        'gender': 1
    },
    {
        'height': 175,
        'age': 16,
        'gender': 0
    }
]

#appendToFile('python/data-file.json', json.dumps(data))

def train():
    w1 = random.uniform(0, 1)*0.2-0.1
    w2 = random.uniform(0, 1)*0.2-0.1
    b = random.uniform(0, 1)*0.2-0.1
    learning_rate = 0.2
    for i in range(10000):
        data_entry = data[0]

        z = (data_entry['height']*w1) + (data_entry['age']*w2) + b
        prediction = sigmoid(z)
        error_cost = cost(prediction, data_entry['gender'])
        
        if(i%1000 == 0):
            print(prediction)

train()
