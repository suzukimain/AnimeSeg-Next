from setuptools import setup, find_packages

setup(
    name="anime_seg_next",
    version="0.1.0",
    description="AnimeSeg-Next: Anime Character Segmentation (Next Generation)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="suzukimain",
    author_email="gt13579552@gmail.com",
    url="https://github.com/suzukimain/AnimeSeg-Next",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "anime_seg>=0.3.8",
        "torch>=2.0.0",
        "numpy",
        "Pillow",
        "huggingface_hub",
        "safetensors",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
