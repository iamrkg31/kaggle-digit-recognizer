from configparser import ConfigParser
import json
import os


class Config(object):
    """Config parser to load and return neural net configurations"""
    def __init__(self, conf_file=None):
        """
        Initialize the class and set the config file property
        """
        self.config = ConfigParser()
        self.conf_file = os.environ.get('CONFIG_FILE', "config/system.config")\
                        if conf_file else os.path.abspath(conf_file)
        if not os.path.isfile(self.conf_file):
            raise Exception("%s : File does not exist..." % self.conf_file)
        self.config.read(self.conf_file)


    def get_config(self, section, item):
        """Returns the property from the config file"""
        json_acceptable_string = self.config.get(section, item).replace("'", "\"")
        return json.loads(json_acceptable_string)


if __name__ == '__main__':
    pass
