import flask
from flask import Flask,jsonify,request
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TOXIC_BERT_TOKENIZER = AutoTokenizer.from_pretrained("unitary/toxic-bert")
TOXIC_BERT = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")

EMOTION_BERT_TOKENIZER= AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-emotion-analysis")
EMOTION_BERT= AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-emotion-analysis")

STARS_BERT_TOKENIZER = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
STARS_BERT = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

FINANCIAL_BERT_TOKENIZER = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
FINANCIAL_BERT = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
app=Flask(__name__)
@app.route("/toxic_bert",methods=['POST'])
def handle_toxicBert():
    text=TOXIC_BERT_TOKENIZER.encode(request.json.get('text'),return_tensors='pt')
    result=(((TOXIC_BERT(text).logits.softmax(dim=1))*100).round()/100)[0]
    return flask.jsonify({'toxic':result[0].item(),'obscene':result[1].item(),'insult':result[2].item(),'identity_hate':result[3].item(),'threat':result[4].item(),'severe_toxic':result[5].item()})
@app.route("/emotion_bert",methods=['POST'])
def handle_emotionBert():
    text = EMOTION_BERT_TOKENIZER.encode(request.json.get('text'), return_tensors='pt')
    result = (((EMOTION_BERT(text).logits.softmax(dim=1)) * 100).round() / 100)[0]
    label_map = EMOTION_BERT.config.id2label
    response = {label_map[i]: result[i].item() for i in range(len(result))}
    return jsonify(response)

@app.route("/stars_bert",methods=['POST'])
def handle_starsBert():
    text = STARS_BERT_TOKENIZER.encode(request.json.get('text'), return_tensors='pt')
    result = (((STARS_BERT(text).logits.softmax(dim=1)) * 100).round() / 100)[0]
    label_map = STARS_BERT.config.id2label
    response = {label_map[i]: result[i].item() for i in range(len(result))}
    return jsonify(response)

@app.route("/financial_bert",methods=['POST'])
def handle_financialBert():
    text = FINANCIAL_BERT_TOKENIZER.encode(request.json.get('text'), return_tensors='pt')
    result = (((FINANCIAL_BERT(text).logits.softmax(dim=1)) * 100).round() / 100)[0]
    label_map = FINANCIAL_BERT.config.id2label
    response = {label_map[i]: result[i].item() for i in range(len(result))}
    return jsonify(response)
@app.route("/")
def handle_index():
    return jsonify({'name':'Model inference api','available_models':'none'})

app.run()

