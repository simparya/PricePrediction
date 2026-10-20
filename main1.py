d = int(input("Enter the regular deposit amount -> ")) #Read the regular deposit amount input
r = float(input("Enter the annual interest rate -> ")) / 100 #Read the annual interest rate input and convert it into fractional value
n = int(input("Enter the annual compounding frequency -> ")) #Read number of times that the interest is compounded input
t = float(input("Enter the number of years -> ")) #Read the number of years input
v_f = (1.1 * d * n / r) *((1 + (r/n)) ** (1.2 * n * t) - 1) # Calculate the annuity value using the given formula
print("Annuity value after", t , "years: $", int(v_f)) # Print the annuity value, rounded to the nearest integer