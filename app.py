from fastai.vision.all import *
import gradio as gr

learn = load_learner('model.pkl')

categories = ('Headphones', 'Neckband', 'TWS')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(type='pil', label="Uploaded Image", source="webcam", shape=(192,192))

label = gr.outputs.Label()
examples = ['headphones.jpg', 'neckband.jpg', 'tws.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)
