from taipy.gui import Gui
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import numpy as np
from PIL import Image as images

model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(10))

model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=[SparseCategoricalAccuracy()])

model.save('baseline.keras')

def predict_image(model, path_to_img):
    img = images.open(path_to_img)
    img = img.convert("RGB")
    img = img.resize((32, 32))
    data = np.asarray(img)
    data = data / 255.0
    data = data.reshape((1, 32, 32, 3))
    probs = model.predict(data)
    top_prob = probs.max()
    top_pred = probs.argmax()
    class_names = ['bee', 'elephant', 'giraffe', 'koala', 'orangutan', 'panda', 'panther', 'penguen', 'polar bear', 'turtle', 'rhino']
    top_pred = class_names[top_pred]
    return top_prob, top_pred

content = ""
img_path = "placeholder_image.png"
prob = 0
pred = ""

index = """
<|text-center|
<|{"logo.png"}|image|width=25vw|>

<|{content}|file_selector|extensions=.png|>
select an image from your file system

<|{pred}|>

<|{img_path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>
>
"""

def on_change(state, var_name, var_val):
    if var_name == "content":
        top_prob, top_pred = predict_image(model, var_val)
        state.prob = round(top_prob * 100)
        state.pred = "This is a " + top_pred
        state.img_path = var_val  
        app.refresh()  

app = Gui(page=index)

if __name__ == "__main__":
    app.run(use_reloader=True)
