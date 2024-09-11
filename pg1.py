def factorial(x):
    if x==1:
        return x
    else:
        return x*factorial(x-1)
    print("linear recursion \n")
x=int(input("enter the number:"))
if x<0:
        print("invalid number")     
elif x==0:
    print("fact of zero is one")
else:
        print("factorial of",x,"is",factorial(x))