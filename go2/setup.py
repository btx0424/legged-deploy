import sys
import logging
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CMakeBuild(build_ext):
    def run(self):

        build_temp = Path(self.build_temp) 
        if not build_temp.exists():
            build_temp.mkdir(parents=True)
        
        sourcedir = Path(".").resolve()
        subprocess.run(["cmake", sourcedir], cwd=build_temp)
        subprocess.run(["cmake", "--build", "."], cwd=build_temp)

        # # generate stubs
        logging.debug("Generating stubs...")
        subprocess.run(["pybind11-stubgen", "go2deploy.go2py", "-o", "."], check=True)

setup(
    name="go2deploy",
    version="0.0.1",
    author="btx0424",
    author_email="btx0424@outlook.com",
    ext_modules=[Extension("go2py", [])],
    packages=find_packages("."),
    cmdclass={"build_ext": CMakeBuild},
    install_requires=["numpy", "pybind11-stubgen"],
)
