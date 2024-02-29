from transformers import pipeline
import gradio as gr


modelName = "Chest_Xray"
hfUser = "Hemg"


def prediction_function(inputFile):
    # get user name of their hugging face
    modelPath = hfUser + "/" + modelName
    # takes some time
    classifier = pipeline("image-classification", model=modelPath)

    try:
        result = classifier(inputFile)
        predictions = dict()
        labels = []
        for eachLabel in result:
            predictions[eachLabel["label"]] = eachLabel["score"]
            labels.append(eachLabel["label"])
        result = predictions
    except:
        result = "no data provided!!"

    return result


# change modelName parameter
def create_demo():
    demo = gr.Interface(
        fn=prediction_function,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=3),
    )
    demo.launch()


create_demo()
