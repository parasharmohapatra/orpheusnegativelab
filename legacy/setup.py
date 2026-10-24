from setuptools import setup

APP = ['orpheus.py']
DATA_FILES = []
OPTIONS = {
    'iconfile': 'onl_logo.icns',  # Your .icns file
    'packages': [],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)