from geomfitty._descriptor import Position, Direction, PositiveNumber
from .test_util import assert_vector_equal

import pytest


class DemoClass:
    pos = Position()
    drn = Direction()
    num = PositiveNumber()


class TestPosition:
    def test_construction(self):
        d = DemoClass()
        d.pos = [1, 2, 3]
        assert_vector_equal(d.pos, [1, 2, 3])

    def test_every_instance_has_its_own_private_data(self):
        d1 = DemoClass()
        d2 = DemoClass()
        d1.pos, d2.pos = [0, 0, 0], [1, 1, 1]
        assert_vector_equal(d1.pos, [0, 0, 0])
        assert_vector_equal(d2.pos, [1, 1, 1])

    def test_attribute_error(self):
        d = DemoClass()
        with pytest.raises(AttributeError):
            d.pos


class TestDirection:
    def test_construction(self):
        d = DemoClass()
        d.drn = [1, 0, 0]
        assert_vector_equal(d.drn, [1, 0, 0])

    def test_a_direction_is_normalized_on_construction(self):
        d = DemoClass()
        d.drn = [2, 0, 0]
        assert_vector_equal(d.drn, [1, 0, 0])

    def test_every_instance_has_its_own_private_data(self):
        d1 = DemoClass()
        d2 = DemoClass()
        d1.drn, d2.drn = [1, 0, 0], [0, 1, 0]
        assert_vector_equal(d1.drn, [1, 0, 0])
        assert_vector_equal(d2.drn, [0, 1, 0])

    def test_attribute_error(self):
        d = DemoClass()
        with pytest.raises(AttributeError):
            d.drn


class TestPositiveNumer:
    def test_construction(self):
        d = DemoClass()
        d.num = 1
        assert d.num == 1
        assert isinstance(d.num, float)

    def test_a_positive_number_must_be_initialized_with_a_positive_number(self):
        d = DemoClass()
        with pytest.raises(ValueError):
            d.num = -1

    def test_every_instance_has_its_own_private_data(self):
        d1 = DemoClass()
        d2 = DemoClass()
        d1.num, d2.num = 1, 2
        assert d1.num == 1
        assert d2.num == 2

    def test_attribute_error(self):
        d = DemoClass()
        with pytest.raises(AttributeError):
            d.num
