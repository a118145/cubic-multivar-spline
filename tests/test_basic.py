from cubic_multivar_spline import version, example_function


def test_version_is_string():
    assert isinstance(version, str)


def test_example_function():
    assert example_function(4) == 16

test_example_function()
test_version_is_string()
