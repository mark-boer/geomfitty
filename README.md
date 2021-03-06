# geomfitty
A python library for fitting 3D geometric shapes

[![Build Status](https://travis-ci.org/mark-boer/geomfitty.svg?branch=master)](https://travis-ci.org/mark-boer/geomfitty)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mark-boer/geomfitty/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

## Development
First clone the repository
```
git clone git@github.com:mark-boer/geomfitty.git
cd geomfitty
```

Install the package as in editable mode and install the dev requirements.
```
pip install -e .
pip install -r requirements-dev.txt
```

Run the tests
```
pytest
```

Run the code formatter
```
black .
```

### todo
 - [ ] Add more tests
     - [ ] Cone fit
     - [ ] Fuzz torus
     - [ ] Fuzz Cone
 - [ ] Add remaining fits
     - [x] torus
     - [x] circle3D
     - [ ] Cone
 - [ ] Allow fits to run without initial guess
     - [ ] Cylinder
     - [ ] Circle3D
     - [ ] Torus
 - [x] Add doctests and include these in pytest
 - [ ] Add jacobian to fits
 - [ ] Add typing and start using mypy
 - [x] Add python 3.4 support
 - [ ] Improve precision of Circle3D and Torus fit
 - [ ] test_fuzz_cylinder sometimes fails (multiple minima?)

 - [ ] Future functionality
     - [ ] Add Coordinate transformations
         - [ ] Rotation
         - [ ] Translation
         - [ ] General
     - [ ] Add 2D geometries
         - [ ] Line
         - [ ] Circle
         - [ ] Ellipse
