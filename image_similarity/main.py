import logging
import argparse
import tensroflow as tf

from src.preprocessing import Similarity
from src import load_model, cassandra_api

#############################

        # load all user_id
        # run double for loop to calculate profile similarity
        # end of each for loop write similarity results to cassandra db

        # keyspaces
        # "newnyup@gmail.com"
        # "tlaznd@0801"

#############################

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

def load_image(user_id, image_file):

    input_image = load(image_file)
    input_image = resize(input_image, 224, 224)
    input_image = normalize(input_image)
    
    return user_id, input_image

def get_args():
    parser = argparse.ArgumentParser(description="Train the kc-electra model")

    parser.add_argument(
        "is-cpu", type=bool, default=True, help="Set CPU, default: True"
    )
    # TODO: Set cassandra address
    parser.add_argument(
        "cassandra-address", type=list, help="Set cassandra address"
    )
    # TODO: Set cassandra port
    parser.add_argument(
        "cassandra-port", type=bool, help="Set cassandra port"
    )

    parser.add_argument(
        "batch-size", type=int, default=1, help="Set batch size, default: 1"
    )

    parser.add_argument(
        "nearest_neighbors", type=int, default=10, help="Set num of nearest neighbors, default: 10"
    )

    parser.add_argument(
        "top_k", type=int, default=20, help="Set top k recommendation number, default: 20"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info(f"Setting Cassandra DB, Address:{args.cassandra_address}, Port:{args.cassandra_port}")

    cassandra_obj = cassandra_api.Cassandra(args.cassandra_address, args.cassandra_port)

    device = '/CPU:0' if args.is_cpu else '/GPU:0'
    logging.info(f"Using device {device}")

    all_user_id_image = cassandra_obj.load_all_user()
    image_dataset = tf.dataset.experimental.from_list(all_user_id_image)
    image_dataset = image_dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    VGG = load_model.VGG((224, 224, 3))

    with tf.device(device):
        feature_vector = [[VGG(tensor_image), tensor_user_id] for tensor_user_id, tensor_image in image_dataset]

        prep = Similarity(
            n_components=args.n_components,
            batch_size=args.batch,
            )
        
        # prep.fit_ipca(feature_vector)
        batch_knn = prep.make_batch_knn(feature_vector, args.nearest_neighbors)
        prep.fit_knn(batch_knn, feature_vector, args.top_k)