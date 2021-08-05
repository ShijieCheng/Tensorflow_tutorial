# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 16:17:37 2021

@author: 成世杰
"""

import json

#将json文件写进
test_dict = {
    'version': "1.0",
    'results': 'video',
    'explain': {
        'used': True,
        'details': "this is for josn test",
  }
}

json_str = json.dumps(test_dict, indent=4)
with open('test_data.json', 'w') as json_file:
    json_file.write(json_str)
    
#读取json文件
with open('test_data.json',encoding='utf-8',
          mode='r') as json_file:
          f_read=json_file.read()
          print(f_read)

data=json.loads(f_read)
print(data)
print(type(data))
print(data['explain'])

#再次写进
json_str2=json.dumps(data['explain'],indent=4)
with open('test_data2.json','w') as f:
    f.write(json_str2)




#data=[ { 'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4, 'e' : 5 } ]
##数组编码成json格式
#data2=json.dumps(data)
#print(data2)
##使用参数让json数据格式化输出
#data={ 'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4, 'e' : 5 }
#data3=json.dumps(data,
#                 sort_keys=True,
#                 indent=4,
#                 separators=(',',':'))
#print(data3)
##json.loads 用于解码 JSON 数据。该函数返回 Python 字段的数据类型。
#jsonData='{"a":1,"b":2,"c":3,"d":4,"e":5}'
#text=json.loads(jsonData)
#print(text)