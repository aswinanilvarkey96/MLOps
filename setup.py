from setuptools import find_packages, setup

version = "2.0.0"
with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

requirements = [    
    'click==8.0.3',
    'hydra-core==1.1.1',
    'matplotlib==3.5.0',
    'numpy==1.19.2',
    'python-dotenv==0.19.2',
    'pytorch_lightning==1.5.8',
    'setuptools==60.2.0',
    'torch==1.10.1',
    'torchvision==0.11.2',
]

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="An introduction to Machine Learning Operations and code organisation.",
    author="Aswin Anil Varkey",
    license="",

    url='https://github.com/cookiecutter/cookiecutter',
    package_dir={'cookiecutter': 'cookiecutter'},
    entry_points={'console_scripts': ['cookiecutter = cookiecutter.__main__:main']},
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=requirements,
    zip_safe=False,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Programming Language :: Python",
        "Topic :: Software Development",
    ],
    keywords=[
        "cookiecutter",
        "Python",
        "projects",
        "project templates",
        "Jinja2",
        "skeleton",
        "scaffolding",
        "project directory",
        "package",
        "packaging",
    ],
)
