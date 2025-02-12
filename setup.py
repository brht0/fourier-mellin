from setuptools import setup, find_packages
import os
import io

this_directory = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='fourier_mellin',
    version='0.1.2',
    author='brht0',
    author_email='todo.todo@todo.com',
    description='Todo',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="todo",
    py_modules=["fourier_mellin"],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'opencv-python',
    ],
    include_package_data=True,
    zip_safe=False,
)
