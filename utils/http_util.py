import json

import requests

from app_config import app_config


def step(instance, nodeIndex):
    url = 'http://%s:%s/step/%s' % (
        app_config.get('env.host'), app_config.get('env.port'), nodeIndex)
    body = instance
    response = requests.post(url, json=body, headers={'Connection': 'close'})
    resource_list = json.loads(response.text)['result']
    return resource_list['ob'], resource_list['rew'], resource_list['done'], resource_list['info']


def get_env_observe():
    url = 'http://%s:%s/env/obv' % (app_config.get('env.host'), app_config.get('env.port'))
    response = requests.get(url)
    resource_list = json.loads(response.text)['result']
    return resource_list


def env_reset():
    url = 'http://%s:%s/env/reset' % (app_config.get('env.host'), app_config.get('env.port'))
    response = requests.get(url)
    success = json.loads(response.text)['success']
    return success


def get_task_list():
    url = 'http://%s:%s/task/list' % (app_config.get('env.host'), app_config.get('env.port'))
    response = requests.get(url)
    tasl_list = json.loads(response.text)['result']
    return tasl_list

def get_hprob():
    url = 'http://%s:%s/env/hprob' % (app_config.get('env.host'), app_config.get('env.port'))
    response = requests.get(url)
    tasl_list = json.loads(response.text)['result']
    return tasl_list

