import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='thyroid_disease_AI',
    url='https://github.com/cilab-ufersa/thyroid_disease_AI',
    author='CILAB',
    author_email='rosana.rego@ufersa.edu.br',
    # Needed to actually package something
    packages=setuptools.find_packages(),
    include_package_data=True,
    # Needed for dependencies
    install_requires=required,
    description='A package to predict thyroid disease',
    long_description=open('README.md').read(),
)