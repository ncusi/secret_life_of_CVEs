#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <extension_to_language.json>

Retrieves languages recognized by github via rest api.
Saves dictionary from file extension to programming language as a json file.
"""
import json
import urllib3
import yaml
import sys


def main():
    extension_to_language_dict_filename = sys.argv[1]

    languages_dict = prepare_languages_dict()
    extension_to_language = prepare_extension_to_language(languages_dict)
    save_extension_to_language_dict(extension_to_language, extension_to_language_dict_filename)


def prepare_languages_dict():
    """
    Retrieves languages recognized by github via rest api
    :return: dictionary with recognized programing languages with connected file extensions
    """
    url = 'https://raw.githubusercontent.com/github/linguist/master/lib/linguist/languages.yml'
    http = urllib3.PoolManager()
    response = http.request("GET", url)
    data = response.data.decode('utf-8')
    languages_dict = yaml.safe_load(data)
    return languages_dict


def prepare_extension_to_language(languages_dict):
    """
    Converts languages dictionary to file extension dictionary
    :param languages_dict: dictionary from programming language to extensions
    :return: dictionary from file extension to list of programming languages
    """
    extension_to_language = {}
    for language, entry in languages_dict.items():
        if entry['type'] == 'programming' and 'extensions' in entry:
            for extension in entry['extensions']:
                if extension not in extension_to_language:
                    extension_to_language[extension] = [language]
                else:
                    extension_to_language[extension].append(language)
    return extension_to_language


def save_extension_to_language_dict(extension_to_language, extension_to_language_dict_filename):
    with open(extension_to_language_dict_filename, 'w') as f:
        json.dump(extension_to_language, f)


if __name__ == '__main__':
    main()
