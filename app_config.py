# coding: utf8

import os

import yaml

# 读取配置文件
CONFIG_FILE = 'D:\work\\config-dev.yaml'


class AppConfig(object):

    def __init__(self, config_file):
        with open(config_file) as f:
            self.yaml_config = yaml.safe_load(f)
            AppConfig.refresh_with_env(self.yaml_config, 'MCOS_')

    # 驼峰转换
    @classmethod
    def parse_to_env_key(cls, key):
        return key.upper().replace('-', '_')

    @classmethod
    def refresh_with_env(cls, conf, prev):
        for k, v in list(conf.items()):
            ek = prev + AppConfig.parse_to_env_key(k)
            if isinstance(v, dict):
                AppConfig.refresh_with_env(v, ek + '_')
            else:
                ev = os.getenv(ek)
                if ev is not None:
                    conf[k] = ev

    def get(self, key, default_value=None):
        key = key.split('.')
        value = self.yaml_config
        for i in key:
            value = value.get(i, None)
            if value is None:
                return default_value
        return value


app_config = AppConfig(CONFIG_FILE)
