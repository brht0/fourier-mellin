from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys
import shutil
import glob

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuildExt(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = 'Debug' if self.debug else 'Release'

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE={cfg}',
            f'-DBUILD_PYTHON_MODULE=ON',
            
            # This is temporary, find something else, or make C++17 compliant
            # f'-DCMAKE_CXX_COMPILER=g++-11',
            # f'-DCMAKE_VERBOSE_MAKEFILE=ON',
        ]

        build_args = ['--config', cfg, '--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''), self.distribution.get_version())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

        # Move .so file to correct place.
        # TODO: Why do I need to be moved?

        built_lib = glob.glob(os.path.join(self.build_temp, '**', 'cv2_fourier_mellin*.so'), recursive=True)[0]
        package_dir = os.path.join(os.path.dirname(__file__), 'cv2_fourier_mellin')
        dest_path = self.get_ext_fullpath(ext.name)
        dest_dir = os.path.dirname(dest_path)
        if not os.path.exists(package_dir):
            os.makedirs(package_dir)
            
        shutil.move(built_lib, os.path.join(dest_dir, "cv2_fourier_mellin.so"))


setup(
    name='cv2_fourier_mellin',
    version='0.1.1',
    author='Todo',
    author_email='todo.todo@todo.com',
    description='A Python package for Fourier Mellin transformation using OpenCV',
    long_description='',
    ext_modules=[CMakeExtension('cv2_fourier_mellin/cv2_fourier_mellin')],
    cmdclass=dict(build_ext=CMakeBuildExt),
    zip_safe=False,
    packages=['cv2_fourier_mellin'],
    install_requires=[
        'numpy',
        'opencv-python',
    ],
    package_data={
        'cv2_fourier_mellin': ['*.so'],
    },
)
