import logging
import argparse
import tensorflow as tf

from src.load_model import VGG
from src.cassandra_api import Cassandra
from src.preprocessing import Similarity


#############################

        # load all user_id
        # run double for loop to calculate profile similarity
        # end of each for loop write similarity results to cassandra db

#############################

# This should be mounted s3 PATH
PATH = "/Users/jaeho/Work/yeoboya/recommender/data/image/"

def load(image_file):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)

    input_image = tf.cast(image, tf.float32)
    
    return input_image

def resize(input_image, height, width):
    input_image = tf.image.resize(
        input_image,
        [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    
    return input_image

def normalize(input_image):
    input_image = (input_image / 127.5) - 1
    
    return input_image

def load_image(image_file):
    id = tf.strings.split(image_file, "/")[-1]
    id = tf.strings.split(id, ".")[0]
    input_image = load(image_file)
    input_image = resize(input_image, 224, 224)
    input_image = normalize(input_image)
    
    return id, input_image

def get_args():
    parser = argparse.ArgumentParser(description="Train the kc-electra model")

    parser.add_argument(
        "--is-cpu", type=bool, default=False, help="Set CPU, default: True"
    )
    # TODO: Set cassandra address
    parser.add_argument(
        "--cassandra-address", type=list, help="Set cassandra address"
    )
    # TODO: Set cassandra port
    parser.add_argument(
        "--cassandra-port", type=bool, help="Set cassandra port"
    )

    parser.add_argument(
        "--batch-size", type=int, default=16, help="Set batch size, default: 1"
    )

    parser.add_argument(
        "--nearest_neighbors", type=int, default=10, help="Set num of nearest neighbors, default: 10"
    )

    parser.add_argument(
        "--top_k", type=int, default=20, help="Set top k recommendation number, default: 20"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info(f"Setting Cassandra DB, Address:{args.cassandra_address}, Port:{args.cassandra_port}")

    device = '/CPU:0' if args.is_cpu else '/GPU:0'
    logging.info(f"Using device {device}")

    img_dataset = tf.data.Dataset.list_files(str(PATH + "*.jpg"))
    img_dataset = img_dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    img_dataset = img_dataset.batch(args.batch_size)
    vgg = VGG()

    with tf.device(device):
        feature_vector = [[tensor_user_id, vgg(tensor_image)] for tensor_user_id, tensor_image in img_dataset]

        # if n_components is larger than batch_sie raise an error
        prep = Similarity(
            n_components=min(100, args.batch_size),
            )
        
        prep.fit_ipca(feature_vector)
        batch_knn = prep.make_batch_knn(feature_vector, args.nearest_neighbors)
        user_profile_similarity = prep.fit_knn(batch_knn, feature_vector, args.top_k)
        
        logging.info("Writing similarity scores to Cassandra")
        cassandra_obj = Cassandra()
        cassandra_obj.write_similarity(user_profile_similarity)