from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, \
    DefaultDataCollator, AutoImageProcessor, pipeline
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from datasets import load_dataset
from huggingface_hub import login, create_repo, get_full_repo_name, upload_file
import numpy as np
import evaluate
import torch
import os


checkpoint = "google/vit-base-patch16-224-in21k"
hfToken = "#"
hfUser = "Hemg"
modelDir = "Chest_Xray"


######################
### Model Training ###
######################

def start_training_computer_vision():
    login(token=hfToken, add_to_git_credential=True)

    # model created in hugging face
    dataset, preTrainedModel, imageProcessor = model_creation()
    train_model(dataset, preTrainedModel, imageProcessor, modelDir)
    

def model_creation():
    dataset = load_dataset(
        "keremberke/chest-xray-classification",
        "mini",
        # data_dir=folderDir,
        split="train"
    )
    dataset = dataset.train_test_split(test_size=0.2)

    labels = dataset["train"].features["labels"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    preTrainedModel = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    imageProcessor = AutoImageProcessor.from_pretrained(checkpoint)

    normalize = Normalize(
        mean=imageProcessor.image_mean,
        std=imageProcessor.image_std
    )

    size = (
        imageProcessor.size["shortest_edge"]
        if "shortest_edge" in imageProcessor.size
        else (imageProcessor.size["height"], imageProcessor.size["width"])
    )

    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    # transformation of image pixel to value
    def transforms(examples):
        examples["pixel_values"] = [_transforms(
            img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    dataset = dataset.with_transform(transforms)
    return dataset, preTrainedModel, imageProcessor


def compute_metrics(evalPred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = evalPred
    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(
        predictions=predictions,
        references=labels
    )


def train_model(dataset, preTrainedModel, imageProcessor, modelDir):
    dataCollator = DefaultDataCollator()

    # setting the hyperparameters
    trainingArgs = TrainingArguments(
        # saving the model locally
        output_dir=modelDir,
        # output_dir="model",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=4e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        # change epochs
        num_train_epochs=4,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
    )

    # settings parameters of trainer
    trainer = Trainer(
        model=preTrainedModel,
        args=trainingArgs,
        data_collator=dataCollator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=imageProcessor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    trainer.push_to_hub()


##################
### Prediction ###
##################

def predict_computer_vision(modelDir, inputFile):
    classifier = pipeline("image-classification", model=modelDir)
    result = classifier(inputFile)

    try:
        result = classifier(inputFile)
    except:
        result = "no data provided!!"

    return result


##############
### Upload ###
##############

# def upload_app_computer_vision(modelDir, title):
#     login(token=hfToken, add_to_git_credential=True)
#     update_app_file(title)
#     create_hf_model(modelDir, title)
#     create_hf_space(title)


# def update_app_file(title):
#     appFilePath = 'app.py'

#     # Open the file in read mode and read lines into a list
#     with open(appFilePath, 'r') as file:
#         lines = file.readlines()
#     # Replace the line with initialization of modelName
#     lines[4] = 'modelName = "' + title + '"\n'

#     # Open the file in write mode and write back the lines
#     with open(appFilePath, 'w') as file:
#         file.writelines(lines)


# def create_hf_model(modelDir, title):
#     savedModel = AutoModelForImageClassification.from_pretrained(modelDir)
#     imageProcessor = AutoImageProcessor.from_pretrained(modelDir, local_files_only=True)
#     trainingArgsFilePath = os.path.join(modelDir, "training_args.bin")
#     trainingArgs = torch.load(trainingArgsFilePath)
#     trainingArgs.hub_model_id = title

#     trainer = Trainer(
#         model=savedModel,
#         tokenizer=imageProcessor,
#         args=trainingArgs,
#     )

#     trainer.push_to_hub()


# def create_hf_space(targetSpaceName):
#     create_repo(repo_id=targetSpaceName, token=hfToken, repo_type="space", space_sdk="gradio")
#     repo_name = get_full_repo_name(model_id=targetSpaceName, token=hfToken)

#     upload_file(
#         path_or_fileobj='app.py',
#         path_in_repo="app.py",
#         repo_id=repo_name,
#         repo_type="space",
#         token=hfToken,
#     )

#     upload_file(
#         path_or_fileobj='requirements.txt',
#         path_in_repo="requirements.txt",
#         repo_id=repo_name,
#         repo_type="space",
#         token=hfToken,
#     )


start_training_computer_vision()
inputFile = "D:/computer_vision/image.jpg"
print(predict_computer_vision(modelDir, inputFile))