class Inherited:

    def foo(self) -> self:
        return self

test = Inherited().foo()
