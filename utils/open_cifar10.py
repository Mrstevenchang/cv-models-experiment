import numpy as np
import cv2

filepath = '../dataset/cifar-10-batches-py/'


def unpickle(file):
    import pickle
    with open(filepath + file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# data_batch_1 = unpickle('data_batch_1')
#
# cifar_data_1_label = data_batch_1[b'labels']
# cifar_data_1_data = np.array(data_batch_1[b'data'])
# cifar_data_1_data = np.array(cifar_data_1_data)

label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def untar_cifar10(filename):
    data_batch = unpickle(filename)
    data = np.array(data_batch[b'data'])
    label = np.array(data_batch[b'labels'])
    for i in range(10000):
        image = data[i]
        image = image.reshape(-1, 1024)
        r = image[0, :].reshape(32, 32)
        g = image[1, :].reshape(32, 32)
        b = image[2, :].reshape(32, 32)
        img = np.zeros((32, 32, 3))
        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
        cv2.imwrite('../dataset/' + filename + '/cifar10_' + str(i) + '_' + str(label_name[label[i]]) + '.jpg', img)


untar_cifar10('data_batch_1')

