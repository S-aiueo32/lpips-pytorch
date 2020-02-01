from setuptools import setup

setup(
    name='lpips_pytorch',
    version='latest',
    description='LPIPS as a Package.',
    packages=['lpips_pytorch', 'lpips_pytorch.modules'],
    author='So Uchida',
    author_email='s.aiueo32@gmail.com',
    install_requires=["torch", "torchvision"],
    url='https://github.com/S-aiueo32/lpips-pytorch',
)
