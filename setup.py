from setuptools import setup

setup(
    name='pass_logit',
    version='0.1',
    description='Implementation of PASS-GLM for logistic regression',
    url='http://github.com/amacfie/pass_logit',
    author='Andrew MacFie',
    author_email='andrew222651@fastmail.com',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    packages=['pass_logit'],
    install_requires=['pyspark', 'numpy', 'sympy', 'theano'],
    zip_safe=False,
)
