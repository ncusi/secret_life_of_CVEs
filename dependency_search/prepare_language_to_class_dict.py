#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <language_to_class.json>

Saves dictionaries from programming language to language classes as a json file.
"""
import json
import sys


def main():
    language_to_class_dict_filename = sys.argv[1]

    languages_classes = prepare_languages_classes()
    language_to_class = prepare_language_to_class(languages_classes)
    save_language_to_class(language_to_class, language_to_class_dict_filename)


def prepare_languages_classes():
    """
    Prepares a list of programming language classes
    :return: list (language name column, programming paradigm, compilation class, type class, memory model
    """
    languages = [('lang_Ada', 1, 1, 1, 1),
                 ('lang_Assembly', 1, 1, 2, 2),
                 ('lang_C', 1, 1, 2, 2),
                 ('lang_C#', 1, 1, 1, 1),
                 ('lang_C++', 1, 1, 2, 1),
                 ('lang_Clojure', 3, 2, 1, 1),
                 ('lang_Common Lisp', 3, 2, 1, 1),
                 ('lang_Crystal', 1, 1, 1, 1),
                 ('lang_Dart', 1, 1, 1, 1),
                 ('lang_Elixir', 3, 2, 1, 1),
                 ('lang_Emacs Lisp', 3, 2, 1, 1),
                 ('lang_Erlang', 3, 2, 1, 1),
                 ('lang_Forth', 1, 2, 2, 2),
                 ('lang_Go', 1, 1, 1, 1),
                 ('lang_Groovy', 1, 2, 1, 1),
                 ('lang_Haskell', 3, 1, 1, 1),
                 ('lang_Io', 1, 2, 1, 1),
                 ('lang_Java', 1, 1, 1, 1),
                 ('lang_JavaScript', 1, 2, 2, 1),
                 ('lang_Kotlin', 1, 1, 1, 1),
                 ('lang_Lua', 2, 2, 2, 1),
                 ('lang_Objective-C,', 1, 1, 2, 1),
                 ('lang_OCaml', 3, 1, 1, 1),
                 ('lang_Perl', 1, 2, 2, 1),
                 ('lang_PHP', 1, 2, 2, 1),
                 ('lang_PLSQL', 1, 1, 1, 1),
                 ('lang_PowerShell', 2, 2, 1, 1),
                 ('lang_Prolog', 3, 2, 1, 1),
                 ('lang_Python', 1, 2, 1, 1),
                 ('lang_R', 1, 2, 1, 1),
                 ('lang_Ruby', 1, 2, 1, 1),
                 ('lang_Scala', 3, 1, 1, 1),
                 ('lang_Scheme', 3, 2, 1, 1),
                 ('lang_Smalltalk', 1, 2, 1, 1),
                 ('lang_Solidity', 1, 1, 1, 2),
                 ('lang_Swift', 1, 1, 1, 1),
                 ('lang_Tcl', 1, 2, 2, 1),
                 ('lang_TypeScript', 1, 1, 1, 1),
                 ('lang_VBScript', 2, 1, 1, 1)]
    return languages


def prepare_language_to_class(languages):
    """
    Converts list of language classes to dictionaries
    :param languages: list of programming language classes
    :return: dictionaries from language to each class
    """
    language_programming_paradigm = {}
    language_compilation_class = {}
    language_type_class = {}
    language_memory_model = {}
    for entry in languages:
        language = entry[0]
        language_programming_paradigm[language] = entry[1]
        language_compilation_class[language] = entry[2]
        language_type_class[language] = entry[3]
        language_memory_model[language] = entry[4]
    language_to_class = {
        'programming_paradigm': language_programming_paradigm,
        'compilation_class': language_compilation_class,
        'type_class': language_type_class,
        'memory_model': language_memory_model
    }
    return language_to_class


def save_language_to_class(language_to_class, language_to_class_dict_filename):
    with open(language_to_class_dict_filename, 'w') as f:
        json.dump(language_to_class, f)


if __name__ == '__main__':
    main()
