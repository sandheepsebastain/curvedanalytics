from worldbankapp import app
import json, plotly
from flask import render_template,request
from wrangling_scripts.wrangle_data import return_figures
from AIChatBot.chatBotPredictorTensorFlow import getResponse
@app.route('/')
@app.route('/index')
def index():

    figures = return_figures()

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html',
                           ids=ids,
                           figuresJSON=figuresJSON)
@app.route('/api/chatbotai/<string:Chat_Message>',methods=['GET','POST'])
def APIChatBotAI(Chat_Message):
    if request.method=='GET':
        szMessage=getResponse(Chat_Message)
        return "<h2>"+szMessage+"</h2>"
    if request.method=='POST':
        jsonData = request.get_json()
        return jsonData