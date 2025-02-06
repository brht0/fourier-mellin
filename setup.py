from setuptools import setup, find_packages
setup(
    name='fourier_mellin',
    version='0.1.2',
    author='brht0',
    author_email='todo.todo@todo.com',
    description='Todo',
    long_description=open("README.md").read()
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
