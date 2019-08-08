import numpy

#our data
#[height, age, gender(0=male, 1=female)]
data = [
    [180, 21, 0],
    [169, 17, 1],
    [153, 13, 1],
    [176, 20, 0],
    [172, 35, 1],
    [195, 26, 0],
    [160, 16, 1],
    [180, 41, 0]
]

# sigmoid function (activiation function)
def sigmoid(x):
    return (1 / (1 + numpy.exp(-x)))

#derivative of sigmoid function
def d_sigmoid(x):
    return (sigmoid(x)*(1 - sigmoid(x)))

#derivative of cost function (prediction - target) ** 2
def d_cost(prediction, target):
    return (2 * (prediction - target))

#train the computer giving it our data
def train():
    #initially pick random numbers for our weights and bias
    w1 = numpy.random.randn()*0.1
    w2 = numpy.random.randn()*0.1
    b = numpy.random.randn()*0.1

    #how fast we want the computer to learn
    learning_rate = 0.15
    for i in range(20000):
        #pick a random data entry
        randomDataEntry = data[numpy.random.randint(len(data))]
        
        # apply weights to data entry
        z = randomDataEntry[0]*w1 + randomDataEntry[1]*w2 + b
        
        #get prediction and target
        prediction = sigmoid(z)
        target = randomDataEntry[2]
        
        # -- DERIVATIVE CALCULATIONS --

        #get derivative cost error
        dcost = d_cost(prediction, target)
        
        #get derivative prediction
        d_prediction = d_sigmoid(z)
        
        #derivative of our points with respect to weights (including bias)
        d_w1 = randomDataEntry[0]
        d_w2 = randomDataEntry[1]
        d_b = 1

        #derivative cost with respect to the derivative points
        dcost_dw1 = dcost * d_prediction * d_w1
        dcost_dw2 = dcost * d_prediction * d_w2
        dcost_db = dcost * d_prediction * d_b
        
        #update weights and biases
        w1 -= dcost_dw1 * learning_rate
        w2 -= dcost_dw2 * learning_rate
        b -= dcost_db * learning_rate

    #finally we return our adjusted weights and bias
    return [w1, w2, b]

#test prediction on
learned = train()

#data on female so our prediction should be close to 1
z = 160*learned[0] + 21*learned[1] + learned[2]
prediction = sigmoid(z)
print(prediction)
