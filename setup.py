from setuptools import setup, find_packages

setup(
    name='lpips-pytorch',
    version='latest',
    description='LPIPS as a Package.',
    packages=find_packages(
        exclude=('tests', 'data', 'PerceptualSimilarity')),
    author='So Uchida',
    author_email='s.aiueo32@gmail.com',
    install_requires=["torch", "torchvision"],
    url='https://github.com/S-aiueo32/lpips-pytorch',
)