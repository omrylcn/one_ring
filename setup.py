from setuptools import setup

requirements = [
    "albumentations",
    "datasets",
    "matplotlib",
    "numpy",
    "omegaconf",
    "onnxruntime", 
    "tensorflow",
    "tf2onnx",
    "transformers",
]

setup(
    name="one_ring",
    version="0.0.1",
    description="One tool to rule'em all, one tool to find them. One tool to bring'em all, and in the darkness bind them.",
    author="Omer Yalcin",
    author_email=" omeryalcin48@gmail.com",
    url="https://github.com/omrylcn/one_ring",
    packages=["one_ring"],
    install_requires=requirements,
)
