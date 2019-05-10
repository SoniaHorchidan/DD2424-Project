from network import Network
import datetime


model_num = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
model_name = "model" + str(model_num) + ".h5"

net = Network()
model = net.train(epochs = 1, batch_size = 96)
model.save(model_name)