import os
import sys
import boto3
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Activation

os.environ['AWS_ACCESS_KEY_ID'] = 'AKIA5BCATYXOPMIS6VGF'
os.environ['AWS_SECRET_ACCESS_KEY'] = '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'


def VGG(model_input_shape=(224, 224, 3)):
    object_name = "vgg_face_weights.h5"

    if object_name not in os.listdir(dir_path):
        bucket_name = "simkoong-weight"

        # Download image from S3 and save locally
        s3 = boto3.resource('s3', region_name='ap-northeast-2')  # Replace 'us-east-1' with your desired region
        bucket = s3.Bucket(bucket_name)
        object = bucket.Object(object_name)

        # Download the image and save it locally
        object.download_file(os.path.join(dir_path,object_name))

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=model_input_shape))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    model.load_weights(os.path.join(dir_path,object_name))

    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    return vgg_face_descriptor