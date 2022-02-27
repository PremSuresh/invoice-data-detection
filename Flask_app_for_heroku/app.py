import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask
from flask_restful import reqparse, Api, Resource
from keras import backend as K
from keras.engine import Layer
from keras.models import load_model
import io 
import re
import os
from os import listdir
from os.path import isfile, join
from google.cloud import vision
from google.cloud import storage
from google.protobuf import json_format
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
app = Flask(__name__)
api = Api(app)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="vision_key.json"
import detect as det
import uploadfiletovision as sendtovision

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2',
                               trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights +=K.tf.trainable_variables(scope=
                                                           "^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                           as_dict=True,
                           signature='default',
                           )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)



strList = ['label_address', 'label_amount', 'label_bill_period',
           'label_cin', 'label_date', 'label_description', 'label_email',
           'label_intents_OrgGST', 'label_intents_VendorGST',
           'label_none', 'label_pan', 'label_ph_no', 'label_quantity', 'label_sac',
           'label_tax_amount', 'label_tax_percent', 'label_tot_in_words',
           'label_total_amount', 'label_vendor_name']


model=load_model("halfnonehalfsac.h5", custom_objects={'ElmoEmbeddingLayer': ElmoEmbeddingLayer()})
model._make_predict_function()
parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictClass(Resource):
	def get(self):
		file_name='uploads/'
		print("enter the file name")
		args = parser.parse_args()
		user_query = args['query']
		name=user_query
		mypath='/'
		file_path=mypath+name
		file_name=file_name+name
		path='gs://actionboard-219211-vcm/uploads/'+name
		json_string=det.async_detect_document(path, r'gs://actionboard-219211-vcm/dest/')
		final=[]
		source=[]

		response = json_format.Parse(json_string, vision.types.AnnotateFileResponse())

		# The actual response for the first page of the input file.
		x=[]
		for j in range(0,len(response.responses)):
		    first_page_response = response.responses[j]
		    annotation = first_page_response.full_text_annotation
		    y=annotation.text.split("\n")
		    x.append(y)
		for invoice in x:
		    while '' in invoice:
		        invoice.remove('')
		for k in x:
		    for l in k:
		        final.append(l)
		        source.append(name)
		Tot_in_words=[]
		Amounts=[]
		Amounts_conf=[]
		Amounts_name=[]
		Vendor_gst=[]
		Org_gst=[]
		cin=[]
		for i in final:
		    check_cin=re.search(r"[a-zA-Z0-9%]{21}",i)
		    if check_cin is None or check_cin.group().isalpha()==True:
		        print('not cin')
		    else:
		        check_cin=check_cin.group()
		        cin.append(check_cin)
		        continue
		    x = re.search(r"[a-zA-Z0-9%]{15}",i)
		    if x is None or x.group().isalpha()==True:
		        y=i
		        y=y.replace(":"," : ")
		        #y=re.sub("\.\d{2}","",y)
		        y=y.lower()
		        a=np.array([y])
		        print(a)
		        output1= model.predict(a)
		        strList1 = ['label_address', 'label_amount', 'label_bill_period',
		       'label_cin', 'label_date', 'label_description', 'label_email',
		        'label_intents_OrgGST', 'label_intents_VendorGST',
		       'label_none', 'label_pan', 'label_ph_no', 'label_quantity', 'label_sac',
		       'label_tax_amount', 'label_tax_percent', 'label_tot_in_words',
		       'label_total_amount', 'label_vendor_name']
		        result1 = zip(output1[0]
		             , strList1)
		        z=set(result1)
		        greatest=0
		        for j in z:
		            if(j[0]>greatest):
		                greatest=j[0]
		                name=j[-1]
		                conf=j[0]
		        print(name)
		        if (name=='label_amount') or (name=='label_tax_amount') or (name=='label_total_amount'):
		#             r=re.search(r"ph",y)
		#             if r is None:
		            Amounts.append(y)
		            Amounts_name.append(name)
		            Amounts_conf.append(conf)
		        elif (name=='label_tot_in_words'):
		            Tot_in_words.append(y)
		    else:
		        y=x.group()
		        
		        if y[-2]==2 or y[-2]=='%' :
		            y = y[:-2] + 'z' + y[-1:]
		        if y[-3]=='i' or y[-3]=='I' :
		            y = y[:-3] + '1' + y[-2:]
		        y=y.lower()
		        a=np.array([y])
		        print(a)
		        output= model.predict(a)
		        strList = ['label_address', 'label_amount', 'label_bill_period',
		       'label_cin', 'label_date', 'label_description', 'label_email',
		        'label_intents_OrgGST', 'label_intents_VendorGST',
		       'label_none', 'label_pan', 'label_ph_no', 'label_quantity', 'label_sac',
		       'label_tax_amount', 'label_tax_percent', 'label_tot_in_words',
		       'label_total_amount', 'label_vendor_name']
		        result = zip(output[0]
		             , strList)
		        z=set(result)
		        greatest=0
		        for j in z:
		            if(j[0]>greatest):
		                if(j[-1]=="label_intents_VendorGST") or (j[-1]=="label_intents_OrgGST"):
		                    greatest=j[0]
		                    name=j[-1]
		                    conf=j[0]
		        print(name)
		        if(name=='label_intents_VendorGST'):
		            clean=x.group()
		            if clean[-2]=='2' or clean[-2]=='%' :
		                idx=-2
		                clean = clean[0:idx] + 'Z' + clean[idx+1:]
		            if clean[-3]=='I' or clean[-2]=='i' :
		                idx=-3
		                clean = clean[0:idx] + '1' + clean[idx+1:]
		            Vendor_gst.append(clean)
		        elif(name=='label_intents_OrgGST'):
		            clean=x.group()
		            if clean[-2]=='2' or clean[-2]=='%' :
		                idx=-2
		                clean = clean[0:idx] + 'Z' + clean[idx+1:]
		            if clean[-2]=='I' or clean[-2]=='i' :
		                idx=-3
		                clean = clean[0:idx] + '1' + clean[idx+1:]
		            Org_gst.append(clean)
		output = {'Tot_in_words': str(Tot_in_words), 'Amounts': str(Amounts), 'Amounts_conf':str(Amounts_conf), 'Amounts_name':str(Amounts_name), 'Vendor_gst':str(Vendor_gst),'Org_gst':str(Org_gst)}
		return output

api.add_resource(PredictClass, '/')


if __name__ == '__main__':
    app.run(debug=True)