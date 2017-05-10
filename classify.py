import sys
import caffe

net_path='/home/shiina/caffe/examples/mnist/lenet.prototxt'
model_path='/home/shiina/caffe/examples/mnist/lenet_iter_10000.caffemodel'

Number={
    0:"0",
    1:"1",
    2:"2",
    3:"3",
    4:"4",
    5:"5",
    6:"6",
    7:"7",
    8:"8",
    9:"9"
}

Input=caffe.io.load_image(sys.argv[1], color=False)
classifier=caffe.Classifier(net_path, model_path)
predictions=classifier.predict([Input])
print 'input image is %s '  %predictions[0].argmax()
for index, prediction in enumerate(predictions[0]):
    print("("+Number[index]+"):")+str(prediction)
