try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Stars and radial velocity data analysis',
    'author': 'Leonardo dos Santos',
    'download_url': 'https://github.com/RogueAstro/keppy',
    'author_email': 'leonardoags@usp.br',
    'version': '0.1.dev',
    'install_requires': ['numpy', 'scipy', 'emcee'],
    'packages': ['keppy'],
    'name': 'keppy'
}

setup(**config)