from network import Network
import datetime

model_num = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
model_name = "model" + str(model_num) + ".h5"

### Add an existing name of a model to continue training
net = Network(model_name)

### Load a model and continue training
net.load()

model = net.train(epochs = 1 , batch_size = 96)