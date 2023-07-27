# import libraries
import numpy as np
import pandas as pd
import pickle	
import nltk, string
import seaborn as sns
import warnings, torch
warnings.filterwarnings('ignore')
from transformers import Trainer, TrainingArguments
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, render_template, request



# function to evaluate test set
def compute_metrics(p):
    print(type(p))
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='weighted')
    precision = precision_score(y_true=labels, y_pred=pred, average='weighted')
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])




# Initialize tokenizer and saved model				  
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')	
print("hghh")
saved_model1 = BertForSequenceClassification.from_pretrained('./saved_model/bert_base_cased',
															 num_labels=6)
print("model loaded successfully")														 

# set training arguments
args = TrainingArguments(output_dir="./output",
                         num_train_epochs=9,
                         warmup_steps=500,
                         evaluation_strategy="steps",
                         per_device_train_batch_size=8,
                         per_device_eval_batch_size=8,
                         weight_decay=0.01,
                         logging_dir='./logs',
                         logging_steps=500,
                         eval_steps=500,
                         run_name='bert-base-cased')



# intialize Trainer 
trainer = Trainer(model=saved_model1,
                  args=args,
                  compute_metrics=compute_metrics)
				  				  
print("saved model loaded")



label_map = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise"}
app = Flask(__name__, static_url_path='/static')


# load index web page
@app.route("/")
def index():
    return render_template('index.html')


# get input from user and return prediction
@app.route("/predict")
def predict():
	text = request.args.get("query")
	encoding1 = tokenizer(text, return_tensors="pt")
	encoding1 = {k: v.to(trainer.model.device) for k,v in encoding1.items()}


	outputs = trainer.model(**encoding1)
	label = torch.argmax(outputs['logits']).item()
	ans = label_map[label]
	print(ans)
	return render_template("prediction.html", output=ans, text=text)




# run the app
if __name__ == "__main__":
    app.run(debug=True, port=4996)