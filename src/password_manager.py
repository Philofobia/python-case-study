master_pwd = input("Enter the password: ")


def view():
    # Open the file in read mode
    # r is for read, rstrip is to remove the new line character
    with open("passwords.txt", "r") as f:
        for line in f.readlines():
            data = line.rstrip()
            user, password = data.split(" | ")
            print("User:", user, "Password:", password)


def add():
    user = input("Enter the username: ")
    password = input("Enter the password: ")

    # Open the file in append mode ( creates the file if it doesn't exist )
    # a is for append
    with open("passwords.txt", "a") as f:
        f.write(user + " | " + password + "\n")


# Main loop
while True:
    mode = input(
        "Would you like to add a new password or view existing ones (view, add), press q to quit? "
    ).lower()

    if mode == "q":
        break

    if mode == "view":
        view()
    elif mode == "add":
        add()
    else:
        print("Invalid mode.")
