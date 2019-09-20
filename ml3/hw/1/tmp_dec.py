def my_decorator(foo):
    def wrapper(arg):
        print(1)
        foo(arg)
        print(2)

    return wrapper

@my_decorator
def foo(param=1):
    return 0

print(foo())
