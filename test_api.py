import requests
import pandas as pd
import numpy as np
import json

#reading test data
data=pd.read_excel('df_test.xlsx')
#converting it into dictionary
data=data.to_dict('records')
#packaging the data dictionary into a new dictionary
data_json={'data':data}

#defining the header info for the api request
headers = {
    'content-type': "application/json",
    'cache-control': "no-cache",
}

#making the api request
r=requests.get(url='http://localhost:5001/predictions',headers=headers,data=json.dumps(data_json))

#getting the json data out
data=r.json()

#displaying the data
print(data)