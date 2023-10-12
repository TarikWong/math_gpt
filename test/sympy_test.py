# -*- coding: utf-8 -*-
# @Time : 2023/10/11 2:20 下午
# @Author : tuo.wang
# @Version : 
# @Function :
import sympy

x = sympy.Symbol('x')  # 符号化变量
y = sympy.Symbol('y')  # 符号化变量
z = sympy.Symbol('z')  # 符号化变量
sympy.init_printing(use_latex=True)  # 输出设置
print("x:", type(x))
print("y:", type(y))
print(x ** 2 + y + z)
print(sympy.latex(x ** 2 + y + z))
# output
# x: <class 'sympy.core.symbol.Symbol'>
# y: <class 'sympy.core.symbol.Symbol'>
# x**2 + y + z
# x^{2} + y + z
