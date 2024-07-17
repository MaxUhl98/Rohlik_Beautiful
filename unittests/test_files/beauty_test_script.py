def complex_function(x):
    for i in range(100):
        k = i
        while k >= 50:
            for j in range(100 - i):
                k -= j
                if i > 1:
                    k += i
                elif k == 0:
                    k = 10
                else:
                    k = k ** j
        n = 0
        while n < 10:
            k *= n
            n += k % 3
        if n <= 10:
            n = k ** 2 if k < x else x
        if n != x:
            n *= x
        x = x ** n
        return x if x > n else n


def slightly_complex_code(a: int, b: int) -> int:
    """Example of complex but understandable code"""
    x = 0
    y = 1
    for i in range(a):
        for j in range(b):
            x += a * b % a - a // b
            if x > 0:
                x = 1
            elif x == 0:
                x = 2
            else:
                x = x ** 2
            for k in range(round(x)):
                y = x ** 3 - 4
                if y < 1:
                    y = k
                else:
                    y *= k
            x += 5
            y -= 4
    for i in range(100):
        x -= 45
        x += i
        y = y ** i
    return x ** y


def do_nothing(a):
    return a


def another_complex_function(x, y):
    result = 0

    if x > 0:
        if y > 0:
            result += x + y
        elif y < 0:
            result -= x + y
        else:
            result *= x
    elif x < 0:
        if y > 0:
            result = x * y
            if result > 100:
                result = 100
            else:
                result = -100
        elif y < 0:
            result = x // y
            if result % 2 == 0:
                result += 2
            else:
                result -= 2
        else:
            result = x - y
            if result > 0:
                for i in range(result):
                    result -= i
            else:
                for i in range(abs(result)):
                    result += i
    else:
        if y == 0:
            result = 42  # the answer to everything
        else:
            result = y

    for i in range(1, 10):
        if result % i == 0:
            result += i
        else:
            result -= i

    while result < 100 and result > -100:
        result += x
        if result > 50:
            result -= y
        else:
            result += y
        if result == 0:
            break

    return result


def mildly_complex_function(x, y):
    if x:
        x = y
        if x <= 1:
            x = 5
            y = x // 5
        if x == y:
            result = x
        else:
            result = x * y
    else:
        result = y if x <= 0 else 1

    for i in range(1, 10):
        if result % i == 0:
            result += i
        else:
            result -= i

    while result < 100 and result > -100:
        result += x
        if result > 50:
            result -= y
        else:
            result += y
        if result == 0:
            break

    return result
