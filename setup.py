from setuptools import setup

setup(
    name='pycebox',
    version='0.0.1',
    description='The Python Individual Conditional Expectation Toolbox',
    url='https://github.com/AustinRochford/PyCEbox',
    author='Austin Rochford',
    author_email='austin.rochford@gmail.com',
    license='MIT',
    packages=['pycebox'],
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'six'
    ]
)
