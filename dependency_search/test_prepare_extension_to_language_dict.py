from unittest import TestCase

from prepare_extension_to_language_dict import prepare_extension_to_language


class Test(TestCase):
    def test_prepare_extension_to_language_on_one_language(self):
        language_dict = {'Assembly': {'type': 'programming',
                                      'color': '#6E4C13',
                                      'aliases': ['asm', 'nasm'],
                                      'extensions': ['.asm', '.a51', '.i', '.inc', '.nas', '.nasm'],
                                      'tm_scope': 'source.assembly',
                                      'ace_mode': 'assembly_x86',
                                      'language_id': 24}
                         }
        extension_to_language_dict = prepare_extension_to_language(language_dict)
        self.assertIn('.asm', extension_to_language_dict)
        self.assertEqual(extension_to_language_dict['.asm'], ['Assembly'])

    def test_prepare_extension_to_language_on_languages_with_same_extension(self):
        language_dict = {'C': {'type': 'programming',
                               'color': '#555555',
                               'extensions': ['.c', '.cats', '.h', '.idc'],
                               'interpreters': ['tcc'],
                               'tm_scope': 'source.c',
                               'ace_mode': 'c_cpp',
                               'codemirror_mode': 'clike',
                               'codemirror_mime_type': 'text/x-csrc',
                               'language_id': 41},
                         'C++': {'type': 'programming',
                                 'tm_scope': 'source.c++',
                                 'ace_mode': 'c_cpp',
                                 'codemirror_mode': 'clike',
                                 'codemirror_mime_type': 'text/x-c++src',
                                 'color': '#f34b7d',
                                 'aliases': ['cpp'],
                                 'extensions': ['.cpp',
                                                '.c++',
                                                '.cc',
                                                '.cp',
                                                '.cxx',
                                                '.h',
                                                '.h++',
                                                '.hh',
                                                '.hpp',
                                                '.hxx',
                                                '.inc',
                                                '.inl',
                                                '.ino',
                                                '.ipp',
                                                '.ixx',
                                                '.re',
                                                '.tcc',
                                                '.tpp'],
                                 'language_id': 43}
                         }
        extension_to_language_dict = prepare_extension_to_language(language_dict)
        self.assertIn('.h', extension_to_language_dict)
        self.assertEqual(extension_to_language_dict['.h'], ['C', 'C++'])
