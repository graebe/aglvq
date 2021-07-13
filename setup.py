from setuptools import setup

setup(name='aglvq',
      version='1.0',
      description='AGLVQ Package',
      url='https://github.com/graebe/aglvq',
      author=["Torben Graeber", "Sebastian Vetter"],
      author_email=['torbengraebergt@gmail.com', 'sebastianvetter@outlook.de'],
      license='',
      packages=['aglvq'],
      install_requires=[
          'keras==2.3.1',
          'tensorflow==2.2.0',
          'matplotlib',
          'numpy',
          'scikit-learn'],
      zip_safe=False)