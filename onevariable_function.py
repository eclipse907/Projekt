from builtins import str
import sympy as sym


class OnevariableFunction:

    string_function = ""

    replacements = {
        '^': '**'

    }

    def __init__(self, value):
        self.string_function = value.replace(" ", "")
        for old, new in self.replacements.items():
            self.string_function = value.replace(old, new)

    def __call__(self, given_value):
        expression = sym.sympify(self.string_function)
        return expression.subs("x", given_value).evalf()

    def __gradient__(self, value):
        gradient_func = sym.diff(sym.sympify(self.string_function), "x")
        return gradient_func.subs("x", value).evalf()









