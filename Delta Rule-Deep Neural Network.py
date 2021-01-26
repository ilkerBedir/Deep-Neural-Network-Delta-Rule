import matplotlib.pyplot as plt
import numpy as np
from math import *

from numpy.core.fromnumeric import shape


# dosyadaki txt leri doğrudan okuma


def bipolar_input(fileName, path):
    arr = np.zeros(63)
    with open(path + fileName, 'r') as filestream:
        for line in filestream:
            arr = np.array(line.split(","))
            arr = arr.astype(np.int)
    return arr


# okuduktan sonra hepsini bir tabloda birleştirme
def tablo_yapma(path):
    arrA = bipolar_input("Font_1_A.txt", path)
    arrB = bipolar_input("Font_1_B.txt", path)
    arrC = bipolar_input("Font_1_C.txt", path)
    arrD = bipolar_input("Font_1_D.txt", path)
    arrE = bipolar_input("Font_1_E.txt", path)
    arrK = bipolar_input("Font_1_K.txt", path)
    arrJ = bipolar_input("Font_1_J.txt", path)
    arrA2 = bipolar_input("Font_2_A.txt", path)
    arrB2 = bipolar_input("Font_2_B.txt", path)
    arrC2 = bipolar_input("Font_2_C.txt", path)
    arrD2 = bipolar_input("Font_2_D.txt", path)
    arrE2 = bipolar_input("Font_2_E.txt", path)
    arrK2 = bipolar_input("Font_2_K.txt", path)
    arrJ2 = bipolar_input("Font_2_J.txt", path)
    arrA3 = bipolar_input("Font_3_A.txt", path)
    arrB3 = bipolar_input("Font_3_B.txt", path)
    arrC3 = bipolar_input("Font_3_C.txt", path)
    arrD3 = bipolar_input("Font_3_D.txt", path)
    arrE3 = bipolar_input("Font_3_E.txt", path)
    arrK3 = bipolar_input("Font_3_K.txt", path)
    arrJ3 = bipolar_input("Font_3_J.txt", path)
    result = np.concatenate((arrA, arrB, arrC, arrD, arrE, arrJ, arrK, arrA2, arrB2, arrC2, arrD2, arrE2, arrJ2, arrK2,
                             arrA3, arrB3, arrC3, arrD3, arrE3, arrJ3, arrK3), axis=0)
    result = np.reshape(result, (-1, 63))
    return result

# bir inputun bir output ile kıyaslanmasını tahmini


def predict(inputs, weights, weightsIndex, threshold, bipolar_flag):
    activation = 0
    for i in range(0, 63):
        activation += weights[weightsIndex][i] * inputs[i]
    if (activation >= threshold):
        return 1.0
    elif bipolar_flag == True:
        if activation < (-1) * threshold:
            return -1
    return 0.0

# veiyi eğitmek için yazılan fonksiyon


def accuracy(characters, weights, threshold, bipolar_flag):

    total_correct_predict = 0
    for i in range(len(characters)):
        for j in range(0, 7):  # num of output neurons
            result = predict(characters[i], weights,
                             j, threshold, bipolar_flag)
            if j == (i % 7):  # means same letter -> 1: correct ** 0, -1: wrong answers
                if result == 1:
                    total_correct_predict += 1
            elif bipolar_flag == True:
                if result != 1:
                    total_correct_predict += 1
            else:
                if result == 0:
                    total_correct_predict += 1
    return total_correct_predict / (7 * len(characters))


def trainWeights(characters, weights, numof_epochs, learning_rate, threshold, bipolar_flag, update_function):
    log_accuracy = [0.0] * numof_epochs

    for epoch in range(numof_epochs):
        totalError = 0.0
        for i in range(len(characters)):
            inputs = characters[i]
            for j in range(7):
                prediction = predict(inputs, weights, j,
                                     threshold, bipolar_flag)

                # 0: perceptron rule
                if update_function == 0:
                    if j == (i % 7):  # only true for A->A, B->B ...
                        target = 1
                    else:
                        if bipolar_flag == True:
                            target = -1
                        else:
                            target = 0
                    inputIndex = 0
                    if prediction != target:
                        # range(0,63) | range(63,63*2)
                        for k in range(0, 63):

                            # update weights
                            weights[j][k] += learning_rate * \
                                target * inputs[inputIndex]
                            inputIndex += 1

                # 1: delta rule
                elif update_function == 1:
                    if j == (i % 7):  # only true for A->A, B->B ...
                        error = 1 - prediction
                    elif bipolar_flag == True:
                        error = -1 - prediction
                    else:
                        error = 0 - prediction
                    totalError += abs(error)
                    inputIndex = 0
                    # range(0,63) | range(63,63*2)
                    for k in range(0, 63):
                        # update weights
                        weights[j][k] += learning_rate * \
                            error * inputs[inputIndex]
                        inputIndex += 1

        log_accuracy[epoch] = accuracy(
            characters, weights, threshold, bipolar_flag)
        #print('epoch:', epoch + 1, 'current accuracy:', log_accuracy[epoch])

        if log_accuracy[epoch] == 1.0:  # if we reach 100% success rate we can stop
            count = 0  # resize the log_accuracy list with removing zeros
            for i in range(len(log_accuracy)):
                if log_accuracy[i] != 0.0:
                    count += 1
            print('count:', count)
            log_accuracy_no_zeros = [0.0] * count
            for i in range(count):
                log_accuracy_no_zeros[i] = log_accuracy[i]
            return weights, log_accuracy_no_zeros

    return weights, log_accuracy


def binary_table(veri):
    for i in range(0, 21):
        for j in range(0, 63):
            if(veri[i][j] == -1):
                veri[i][j] = 0
    return veri


path = 'liste/'
characters = tablo_yapma(path)
veri = binary_table(characters)

weights = np.zeros(shape=(7, 63))
weights1 = np.zeros(shape=(7, 63))
# weights, bias, totalErrorLog = trainWeights(
#   characters, weights, bias, 125, 0.01, 0.5, True)
weights1, totalErrorLog1 = trainWeights(
    characters, weights1, 100, 0.01, 0.7, True, 0)


#print('bipolar total_error_log: ', totalErrorLog)


#print('bipolar last_weights: ', weights)

#print('bipolar last_bias: ', bias)

print('accurancy: ', totalErrorLog1)

#print('binary last_weights: ', weights1)

#print('binary last_bias: ', bias1)


def test(characters, weights, threshold, bipolar_flag):
    if bipolar_flag == False:
        characters = binary_table(characters)
    error = 0
    for j in range(0, 21):
        for k in range(0, 7):
            c = 0
            out = 0
            for i in range(0, 63):
                out = out + characters[j][i]*weights[k][i]
            if(out > threshold and k == (j % 7)):
                out = 1
            elif(out < threshold and k != (j % 7)):
                out = -1
            else:
                error = error+1
    return error


test_basarı = np.zeros(64)

test_veri = np.zeros(shape=(21, 63))
test_veri = tablo_yapma('test/')
test_veri = binary_table(test_veri)
error = 0.0
error = test(test_veri, weights1, 0.7, False)
test_basarı[0] = 100*(1-(error/(147)))
for i in range(0, 63):
    for j in range(0, 21):
        if(test_veri[j][i] == 0):
            test_veri[j][i] = 1
        else:
            test_veri[j][i] = 0
    error = test(test_veri, weights1, 0.7, False)
    test_basarı[i+1] = 100*(1-(error/(147)))
plt.plot(test_basarı)
plt.xlabel('Değiştirilen Bit Sayısı')
plt.ylabel('Testin Başarı oranı')
plt.title('Binary Veri Perceptron Accurancy-Değiştirlen Bit Grafiği')
plt.show()
