import json
import requests


def post_request():
    warn_ao = {
        "areaCode": "330101",  # 行政区划代码
        "areaName": "区县",  # 行政区划名称
        "content": "物联监测报警",
        "describe": "物联监测报警",
        "location": "{\"type\":\"Point\",\"coordinates\":[121.3322848,28.2716701]}",  # 位置信息
        "sceneType": "内涝",
        "source": "物联监测",
        "type": "积水",
        "sourceRemark": "{\"facilityId\":\"22285\"}"
    }

    security = {
        "securityWarnAO": warn_ao,
        "depth": 4,
        "facilityId": "22285",
        "waterlLevelEnum": "IV级"
    }

    body = json.dumps(security, ensure_ascii=False).encode('utf-8')

    url = 'http://126.1.1.47:6688/water/security/warn/createNewSecWarnWaterl'
    headers = {'Content-Type': 'application/json'}
    r = requests.post(url, data=body, headers=headers)
    if r.status_code != 200:
        print('Post failed')
        print(r.text)
        raise Exception('Post risk failed, status code: {}'.format(r.status_code))
    else:
        print('Post succeed')
