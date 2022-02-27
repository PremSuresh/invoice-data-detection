from flask import Flask
from flask_restful import reqparse, Api, Resource
import io 
import re
import os
from os import listdir
from os.path import isfile, join
app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()
parser.add_argument('query')
global serverstore
serverstore=[]

class PredictClass(Resource):
	def __init__(self):
		self.server=serverstore
	def get(self):
		args = parser.parse_args()
		user_query = args['query']
		self.server.append(user_query)
		serverstore.append(user_query)
		output = {'Response': user_query ,'BucketSoFar': self.server}
		print(output)
		return output

api.add_resource(PredictClass, '/')


if __name__ == '__main__':
    app.run(debug=True)