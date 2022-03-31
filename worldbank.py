from worldbankapp import app
#Comment below line before merging to github which would deploy it to heroku
#Uncomment below line to run on DEV
app.run('0.0.0.0',debug=True,port=3000,threaded=True)