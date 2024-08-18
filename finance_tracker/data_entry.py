from datetime import datetime

DATE_FORMAT = "%d-%m-%Y"
CATEGORIES = {"I": "Income", "E": "Expense"}

# The prompt is the message to display to the user
# The allow_default argument is used to allow the user to skip the input
# strptime() method is used to convert the string to a datetime object
# strftime() method is used to convert the datetime object to a string


def get_date(prompt, allow_default=False):
    date_str = input(prompt)
    if allow_default and not date_str:
        return datetime.today().strftime(DATE_FORMAT)

    try:
        return datetime.strptime(date_str, DATE_FORMAT).strftime(DATE_FORMAT)
    except ValueError:
        return get_date(prompt)


def get_amount():
    try:
        amount = float(input("Enter the amount: "))
        if amount < 0:
            raise ValueError("Amount cannot be negative or zero")
        return amount
    except ValueError as e:
        print(e)
        return get_amount()


def get_category():
    category = input(
        "Enter the category ('I' for Income, 'E' for Expense): ").upper()
    if category not in CATEGORIES:
        print("Invalid category. Please enter 'I' for Income or 'E' for Expense.")
        return get_category()
    return CATEGORIES[category]


def get_description():
    return input("Enter a description (optional): ")
