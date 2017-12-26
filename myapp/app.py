from flask import Flask, render_template, request
from werkzeug import secure_filename
from PIL import Image, ImageFilter
import tensorflow as tf
from cassandra.cluster import Cluster
import datetime


app = Flask(__name__)

@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader_file():
    if request.method == 'POST':
        f = request.files['file']
        createKeySpace()
        # 备注 createKeySpace() 就可以完美运行
        f.save(secure_filename(f.filename))
        number = get_number(f.filename)
        show_result = 'number:' + str(number)
        insertData(f.filename, number)
        return show_result

'''
database create
'''
def createKeySpace():
    # cluster = Cluster(contact_points=['127.0.0.1'],port=9042)
    cluster = Cluster(contact_points=['172.18.0.2'],port=9042)
    session = cluster.connect()
    try:
        session.execute("CREATE KEYSPACE mykeyspace WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }")
        session.execute("USE mykeyspace")
        session.execute("CREATE TABLE mytable (image blob, number int, time text PRIMARY KEY)")
    except Exception as e:
        print(e)


'''
database save
'''
def insertData(fname, n):
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(fname, "rb") as imageFile :
        f = imageFile.read()
        p = bytearray(f)
    # cluster = Cluster(contact_points=['127.0.0.1'],port=9042)
    cluster = Cluster(contact_points=['172.18.0.2'],port=9042)
    session = cluster.connect('mykeyspace')
    try:
        session.execute(
            "INSERT INTO mytable (image, number, time) VALUES (%s, %s, %s)", (p, n, time)
        )
    except Exception as e:
        print(e)


##-------------------------
## number identify
##-------------------------

def imageprepare(file_name):
    #in terminal 'mogrify -format png *.jpg' convert jpg to png
    im = Image.open(file_name).convert('L')
    # resize the picture size to 28 * 28
    img28 = im.resize((28, 28))
    tv = list(img28.getdata()) #get pixel values
    tva = [ (255 - x) * 1.0 / 255.0 for x in tv]
    return tva

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def get_number(file_name):
    tf.reset_default_graph()
    result = imageprepare(file_name)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    init_op = tf.initialize_all_variables()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "/Users/wentanbai/Desktop/code/myapp/model.ckpt")
        prediction=tf.argmax(y_conv,1)
        predint=prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session=sess)
        return predint[0]

if __name__ == '__main__':
    app.run(host="0.0.0.0")