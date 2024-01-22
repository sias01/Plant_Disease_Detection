from keras.models import Sequential, load_model
import tensorflow as tf
import tensorflow_datasets as tfds

dataset = tfds.ImageFolder("D:/Plant-Disease-Detection/PlantVillage/PlantVillage/train")

#num_classes = dataset.classes
print("Number of classes: ",len(dataset))
#print(num_classes)

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

device ='cpu'

model = load_model("./ft_model_plants.h5") 
def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = tf.max(yb, dim=1)
    return dataset.classes[preds[0].item()]